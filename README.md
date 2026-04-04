# sd-webui-auto-prompt-reason

思考型LLM（Reasoning Models）を使用してStable Diffusionのプロンプトを自動生成・拡張するAUTOMATIC1111 SD WebUI拡張機能です。
Ollama、OpenAI互換API、Google Geminiの3つのプロバイダーに対応し、モデルの思考プロセスを表示しながら高品質なプロンプトを生成します。

本プロジェクトは、メンテナンスが停止していた `sd-webui-decadetw-auto-prompt-llm` をベースに、最新の思考型モデルへの対応と機能を大幅に強化したプロダクション品質の拡張機能です。

## 主な機能

*   **3種類のバックエンド対応**: Ollama (ローカル)、OpenAI互換API (OpenRouter, DeepSeek等)、Google Gemini。
*   **思考プロセスの表示**: LLMがプロンプトを生成する際の「思考（Thinking）」プロセスを折りたたみセクションで表示。
*   **思考深度の制御 (Reasoning Effort)**: 3段階（Low/Medium/High）で生成品質と速度のトレードオフを調整。
*   **トークン・実行時間の表示**: 生成完了後に使用したトークン数と処理時間を表示。
*   **柔軟なプロンプト注入**: 生成したプロンプトを既存のプロンプトに追記（Append）、先頭に追加（Prepend）、または置換（Replace）するモードを選択可能。
*   **インメモリ履歴**: 直近50件の生成履歴をセッションごとに保持。
*   **非ブロッキング処理**: 全てのAPI呼び出しをバックグラウンドスレッドで実行し、UIをフリーズさせません。

## 動作環境

*   Python 3.10以上
*   [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
*   必要なライブラリ: `requests`, `PyYAML` (拡張機能インストール時に自動で導入されます)

## インストール

1.  SD WebUIの `extensions` ディレクトリに移動します。
2.  以下のコマンドでリポジトリをクローンします。
    ```bash
    git clone https://github.com/your-repo/sd-webui-auto-prompt-reason.git
    ```
3.  SD WebUIを再起動します。

## 設定

`config.yaml` を編集することで、デフォルトの挙動を変更できます。

```yaml
default_provider: ollama
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:7b"
openai_compatible:
  base_url: "http://localhost:1234/v1"
  model: "deepseek-reasoner"
gemini:
  model: "gemini-2.0-flash"
reasoning:
  effort: "medium"
  show_thinking: true
history:
  max_size: 50
ui:
  prompt_injection_mode: "append" # append | prepend | replace
```

## 使い方

1.  SD WebUIの「Generation」タブにある「Auto Prompt Reason」セクションを開きます。
2.  「Enable」にチェックを入れます。
3.  使用するプロバイダー（Ollama / OpenAI Compatible / Gemini）を選択します。
4.  モデル名、ベースURL、APIキー（必要な場合）を設定します。
5.  「Reasoning Effort」を選択します。
6.  LLMへの指示（プロンプト）を入力し、通常の「Generate」ボタンを押します。
7.  生成されたプロンプトがSD WebUIのプロンプト欄に反映され、画像生成が開始されます。

## プロバイダー設定

### Ollama (ローカル)
ローカルで動作するOllamaを使用します。
*   **Base URL**: `http://localhost:11434` (デフォルト)
*   **Model**: `qwen2.5:7b`, `deepseek-r1:7b` など
*   **思考深度**: 温度（Temperature）にマップされます (Low=0.8, Medium=0.5, High=0.2)。

### OpenAI互換API
OpenRouter、LM Studio、DeepSeek APIなどを使用可能です。
*   **Base URL**: 各サービスのAPIエンドポイント（例: `https://api.deepseek.com/v1`）
*   **API Key**: 各サービスの発行したキーを入力。
*   **Model**: `deepseek-reasoner` など。

### Google Gemini
Google AI Studioから発行したAPIキーが必要です。
*   **API Key**: `?key=` パラメータとして使用されます。
*   **Model**: `gemini-2.0-flash`, `gemini-1.5-pro` など。
*   **思考深度**: `thinkingConfig` の予算（Tokens）にマップされます (Low=1000, Medium=5000, High=10000)。

## 思考深度 (Reasoning Effort)

*   **Low**: 高速な生成。創造性を重視。
*   **Medium**: バランスの取れた生成。
*   **High**: 深い思考を行い、指示に忠実な高品質なプロンプトを生成。

## プロンプト注入モード (Prompt Injection Modes)

*   **Append (Default)**: SDプロンプト欄の内容の後に、LLM生成結果を追記します。
*   **Prepend**: SDプロンプト欄の内容の前に、LLM生成結果を追加します。
*   **Replace**: SDプロンプト欄の内容を、LLM生成結果で完全に置き換えます。

## プロンプト例

LLMに画像生成の詳細を考えさせるためのプロンプト例です。

*   `Cyberpunk city street at night, neon lights, rainy weather, cinematic lighting`
*   `A mystical forest with glowing plants and a hidden waterfall, 8k resolution, detailed texture`
*   `Portrait of a futuristic samurai, intricate armor design, sunset background, masterwork`

## 開発者向け

テストの実行方法:
```bash
pip install -r requirements.txt
pytest tests/ -v --cov=modules --cov=scripts --cov-report=term-missing
```

## ライセンス

Apache License 2.0

---

# sd-webui-auto-prompt-reason

An AUTOMATIC1111 SD WebUI extension that uses reasoning LLMs to automatically generate and enhance Stable Diffusion prompts.
Supports Ollama, OpenAI-compatible APIs, and Google Gemini, displaying the model's thinking process while generating high-quality prompts.

This project is a production-quality extension based on the abandoned `sd-webui-decadetw-auto-prompt-llm`, with significantly enhanced support for modern reasoning models.

## Features

*   **3 Provider Backends**: Supports Ollama (local), OpenAI-compatible APIs (OpenRouter, DeepSeek, etc.), and Google Gemini.
*   **Thinking Process Display**: Shows the LLM's reasoning process in a collapsible "Thinking" section.
*   **Reasoning Effort Control**: 3-level radio (Low/Medium/High) to balance generation depth and speed.
*   **Token & Time Metrics**: Displays tokens used and processing time after each generation.
*   **Flexible Prompt Injection**: Choose to append, prepend, or replace the existing SD prompt with the generated result.
*   **In-Memory History**: Stores the last 50 generations per session for easy reference.
*   **Non-Blocking UI**: All API calls run on background threads to keep the WebUI responsive.

## Requirements

*   Python 3.10+
*   [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
*   Required libraries: `requests`, `PyYAML` (automatically installed)

## Installation

1.  Navigate to your SD WebUI `extensions` directory.
2.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/sd-webui-auto-prompt-reason.git
    ```
3.  Restart SD WebUI.

## Configuration

You can customize the default behavior by editing `config.yaml`.

```yaml
default_provider: ollama
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:7b"
openai_compatible:
  base_url: "http://localhost:1234/v1"
  model: "deepseek-reasoner"
gemini:
  model: "gemini-2.0-flash"
reasoning:
  effort: "medium"
  show_thinking: true
history:
  max_size: 50
ui:
  prompt_injection_mode: "append" # append | prepend | replace
```

## Usage

1.  Open the "Auto Prompt Reason" section in the SD WebUI "Generation" tab.
2.  Check "Enable".
3.  Select your provider (Ollama / OpenAI Compatible / Gemini).
4.  Configure the model name, base URL, and API key (if required).
5.  Select the "Reasoning Effort".
6.  Enter your instruction for the LLM in the text area and click the main "Generate" button.
7.  The generated prompt will be applied to the SD prompt field, and image generation will start.

## Provider Setup

### Ollama (Local)
Uses a locally running Ollama instance.
*   **Base URL**: `http://localhost:11434` (default)
*   **Model**: e.g., `qwen2.5:7b`, `deepseek-r1:7b`
*   **Reasoning Effort**: Mapped to Temperature (Low=0.8, Medium=0.5, High=0.2).

### OpenAI-Compatible
Works with services like OpenRouter, LM Studio, or DeepSeek API.
*   **Base URL**: The service API endpoint (e.g., `https://api.deepseek.com/v1`).
*   **API Key**: Your API Bearer token.
*   **Model**: e.g., `deepseek-reasoner`.

### Google Gemini
Requires an API key from Google AI Studio.
*   **API Key**: Used as a `?key=` parameter.
*   **Model**: e.g., `gemini-2.0-flash`, `gemini-1.5-pro`.
*   **Reasoning Effort**: Mapped to `thinkingConfig` budget (Low=1000, Medium=5000, High=10000 tokens).

## Reasoning Effort

*   **Low**: Fast generation, favors creativity.
*   **Medium**: Balanced depth and speed.
*   **High**: Deep reasoning for high-quality, instruction-faithful prompts.

## Prompt Injection Modes

*   **Append (Default)**: Adds the LLM output after your existing prompt.
*   **Prepend**: Adds the LLM output before your existing prompt.
*   **Replace**: Completely replaces your prompt with the LLM output.

## Example Prompts

Effective prompts to guide the LLM for image generation:

*   `Cyberpunk city street at night, neon lights, rainy weather, cinematic lighting`
*   `A mystical forest with glowing plants and a hidden waterfall, 8k resolution, detailed texture`
*   `Portrait of a futuristic samurai, intricate armor design, sunset background, masterwork`

## For Developers

To run tests and coverage:
```bash
pip install -r requirements.txt
pytest tests/ -v --cov=modules --cov=scripts --cov-report=term-missing
```

## License

Apache License 2.0

