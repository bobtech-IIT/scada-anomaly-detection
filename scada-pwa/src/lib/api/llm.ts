export type LLMProvider = 'openrouter' | 'openai' | 'gemini' | 'anthropic';

export interface LLMSettings {
  provider: LLMProvider;
  apiKey: string;
  model: string;
  customEndpoint?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export class LLMClient {
  public static getDefaultSettings(): LLMSettings {
    return {
      provider: 'openrouter',
      apiKey: '', // Defaults to empty, indicating the user needs to provide one
      model: 'google/gemma-2-9b-it:free',
    };
  }

  public static getSavedSettings(): LLMSettings {
    if (typeof window === 'undefined') return this.getDefaultSettings();
    
    const saved = localStorage.getItem('scada_llm_settings');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse LLM settings", e);
      }
    }
    return this.getDefaultSettings();
  }

  public static saveSettings(settings: LLMSettings): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem('scada_llm_settings', JSON.stringify(settings));
  }

  /**
   * Generates a completion by calling our local Next.js proxy endpoint.
   */
  public static async chat(
    messages: ChatMessage[],
    settings: LLMSettings = this.getSavedSettings()
  ): Promise<string> {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        provider: settings.provider,
        apiKey: settings.apiKey,
        model: settings.model,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`LLM Error: ${response.statusText} (${errText})`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || data.content || '';
  }

  /**
   * RAG Query helper. Formats top-K chunks as context and performs RAG chat.
   */
  public static async ragQuery(
    query: string,
    contextChunks: { text: string; docName: string }[],
    history: ChatMessage[] = [],
    settings: LLMSettings = this.getSavedSettings()
  ): Promise<string> {
    const formattedContext = contextChunks
      .map((c, i) => `[Source: ${c.docName} | Chunk ${i + 1}]\n${c.text}`)
      .join('\n\n');

    const systemPrompt: ChatMessage = {
      role: 'system',
      content: `You are Aegis-AI, an expert SCADA anomaly analysis and predictive maintenance assistant.
Use the following retrieved context documents to answer the user's query. 
Provide professional, accurate, and detailed engineering insights.
If the context does not contain the answer, use your general knowledge of wind turbine operations, SCADA systems, and engineering standards, but state clearly that it is not in the uploaded documents.

--- RETRIEVED CONTEXT ---
${formattedContext || "No custom organizational manuals found. Answering using general engineering knowledge."}
-------------------------`
    };

    const userMessage: ChatMessage = {
      role: 'user',
      content: query,
    };

    const messages = [systemPrompt, ...history, userMessage];
    return this.chat(messages, settings);
  }
}
