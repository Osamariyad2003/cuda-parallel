import { DEEPSEEK_API_KEY } from '../config/deepseek';

interface DeepseekResponse {
  choices: {
    text: string;
  }[];
}

export async function runCode(code: string): Promise<string> {
  try {
    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'deepseek-coder',
        messages: [
          {
            role: 'user',
            content: `Execute this CUDA code and return the output:\n\n${code}`,
          },
        ],
        temperature: 0.7,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data: DeepseekResponse = await response.json();
    return data.choices[0].text;
  } catch (error) {
    console.error('Error executing code:', error);
    return 'Error executing code. Please try again.';
  }
} 