import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { messages, provider, apiKey, model } = await req.json();

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json({ error: 'Messages are required and must be an array' }, { status: 400 });
    }

    // Set fallback keys from environment variables if present
    const key = apiKey || getFallbackKey(provider);

    // If using OpenRouter, allow empty key for free models (OpenRouter handles free calls with key or public endpoints, but having an API key is best. OpenRouter actually requires an authorization header, even for free models you can pass a placeholder if not set, but a real key or blank is needed)
    if (!key && provider !== 'openrouter') {
      return NextResponse.json({ error: `API Key for ${provider} is missing.` }, { status: 400 });
    }

    switch (provider) {
      case 'openrouter':
        return await handleOpenRouter(messages, model, key);
      case 'openai':
        return await handleOpenAI(messages, model, key);
      case 'gemini':
        return await handleGemini(messages, model, key);
      case 'anthropic':
        return await handleAnthropic(messages, model, key);
      default:
        return NextResponse.json({ error: 'Invalid provider selected' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('API Chat route error:', error);
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

async function handleOpenRouter(messages: any[], model: string, key: string) {
  const url = 'https://openrouter.ai/api/v1/chat/completions';
  
  // OpenRouter allows a placeholder or empty authorization for free endpoints,
  // but it's best to supply Bearer with whatever is available.
  const authHeader = key ? `Bearer ${key}` : `Bearer sk-or-v1-placeholder`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': authHeader,
      'HTTP-Referer': 'https://vercel.com',
      'X-Title': 'SCADA Anomaly Detection PWA',
    },
    body: JSON.stringify({
      model: model || 'google/gemma-2-9b-it:free',
      messages: messages,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter Error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return NextResponse.json(data);
}

async function handleOpenAI(messages: any[], model: string, key: string) {
  const url = 'https://api.openai.com/v1/chat/completions';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: model || 'gpt-4o-mini',
      messages: messages,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI Error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return NextResponse.json(data);
}

async function handleGemini(messages: any[], model: string, key: string) {
  // Use OpenAI compatibility endpoint for Gemini
  const url = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: model || 'gemini-1.5-flash',
      messages: messages,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini Error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return NextResponse.json(data);
}

async function handleAnthropic(messages: any[], model: string, key: string) {
  const url = 'https://api.anthropic.com/v1/messages';

  // Anthropic API separates the system prompt from the messages array
  const systemMessage = messages.find((m) => m.role === 'system');
  const systemPrompt = systemMessage ? systemMessage.content : '';

  const filteredMessages = messages
    .filter((m) => m.role !== 'system')
    .map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    }));

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': key,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: model || 'claude-3-5-sonnet-20240620',
      messages: filteredMessages,
      system: systemPrompt,
      max_tokens: 2000,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic Error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  
  // Format Anthropic output to match OpenAI's output structure for our frontend
  const content = data.content?.[0]?.text || '';
  const openaiCompatibleResponse = {
    choices: [
      {
        message: {
          role: 'assistant',
          content: content,
        },
      },
    ],
  };

  return NextResponse.json(openaiCompatibleResponse);
}
