import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { sampleText, provider, apiKey, model } = await req.json();

    if (!sampleText) {
      return NextResponse.json({ error: 'sampleText is required' }, { status: 400 });
    }

    const key = apiKey || getFallbackKey(provider);
    const authHeader = key ? `Bearer ${key}` : `Bearer sk-or-v1-placeholder`;

    const systemPrompt = `You are a SCADA Data Engineering Agent. Your task is to analyze the provided raw, messy text snippet (first 20-30 lines of a telemetry dataset) and generate a JSON mapping schema to clean it.
The target variables we need are:
1. timestamp (string)
2. rotor_rpm (number)
3. gearbox_temp_c (number)
4. vibration_mm_s (number)
5. wind_speed_ms (number)
6. active_power_kw (number)

You must deduce:
- The delimiter used (comma, tab, semicolon, vertical bar, space).
- Which raw column maps to which target variable.
- If there are header rows to skip.
- If unit conversions are needed (e.g., Active Power in Megawatts 'MW' instead of Kilowatts 'kW', meaning we multiply by 1000).

Return ONLY a valid JSON object with the following structure, with NO extra markdown or formatting outside of the JSON block:
{
  "delimiter": ",", // or "\\t", ";", "|", " "
  "headerRowIndex": 0, // Row index of headers, or -1 if none
  "dataStartRowIndex": 1, // Row index where actual data begins
  "mappings": {
    "timestamp": 0, // column index for timestamp
    "rotor_rpm": 1, // column index for RPM
    "gearbox_temp_c": 2, // column index for temp
    "vibration_mm_s": 3, // column index for vibration
    "wind_speed_ms": 4, // column index for wind speed
    "active_power_kw": 5 // column index for power
  },
  "powerMultiplier": 1, // 1000 if raw power is in MW, otherwise 1
  "tempUnit": "C", // 'F' if Fahrenheit (need to convert to C: (F-32)*5/9)
  "explanation": "Brief explanation of how the schema was mapped."
}`;

    const userPrompt = `Here is the messy raw data sample:
------------------
${sampleText}
------------------`;

    // Query OpenRouter (or selected provider)
    let url = 'https://openrouter.ai/api/v1/chat/completions';
    let headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Authorization': authHeader,
      'HTTP-Referer': 'https://vercel.com',
      'X-Title': 'SCADA Data Cleaner',
    };

    let body: any = {
      model: model || 'google/gemma-2-9b-it:free',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.1 // Low temperature for structured predictability
    };

    // Handle provider specifics if not OpenRouter
    if (provider === 'openai') {
      url = 'https://api.openai.com/v1/chat/completions';
      body.model = model || 'gpt-4o-mini';
    } else if (provider === 'gemini') {
      url = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';
      body.model = model || 'gemini-1.5-flash';
    } else if (provider === 'anthropic') {
      url = 'https://api.anthropic.com/v1/messages';
      headers = {
        'content-type': 'application/json',
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
      };
      body = {
        model: model || 'claude-3-5-sonnet-20240620',
        max_tokens: 1000,
        messages: [{ role: 'user', content: userPrompt }],
        system: systemPrompt,
        temperature: 0.1
      };
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`LLM Schema mapping failed: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    let content = '';

    if (provider === 'anthropic') {
      content = data.content?.[0]?.text || '';
    } else {
      content = data.choices?.[0]?.message?.content || '';
    }

    // Clean JSON block markers if LLM returns them
    content = content.replace(/```json/gi, '').replace(/```/g, '').trim();

    // Parse JSON to verify correctness
    const mappingSchema = JSON.parse(content);
    return NextResponse.json(mappingSchema);

  } catch (error: any) {
    console.error('API Clean route error:', error);
    return NextResponse.json({ error: error.message || 'Internal Server Error' }, { status: 500 });
  }
}

function getFallbackKey(provider: string): string {
  switch (provider) {
    case 'openrouter':
      return process.env.OPENROUTER_API_KEY || '';
    case 'openai':
      return process.env.OPENAI_API_KEY || '';
    case 'gemini':
      return process.env.GEMINI_API_KEY || '';
    case 'anthropic':
      return process.env.ANTHROPIC_API_KEY || '';
    default:
      return '';
  }
}
