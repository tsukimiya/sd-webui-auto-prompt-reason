# ADR 001: OpenAI互換エンドポイントでのストリーミング使用（タイムアウト防止）

## ステータス

承認済み（Accepted）

## コンテキスト

本拡張機能（sd-webui-auto-prompt-reason）は、外部LLM API（Ollama、OpenAI互換、Gemini）へのHTTPリクエストを行う。特に以下の制約が存在する：

- **Cloudflare等のリバースプロキシ**: 多くの公開OllamaサーバーやOpenAI互換APIは、Cloudflareなどのリバースプロキシを経由している
- **100秒タイムアウト制限**: CloudflareのFree/Proプランでは、HTTPリクエストのタイムアウトが100秒に設定されている（[Cloudflare Docs - Limits](https://developers.cloudflare.com/workers/platform/limits/#worker-limits)）
- **Thinkingモデルの処理時間**: 特に`deepseek-r1`、`qwen3.5`などのthinkingモデルでは、思考プロセス（reasoning）に時間がかかり、100秒を超えることが一般的

## 決定

**OpenAI互換エンドポイント（`/v1/chat/completions`）では、`stream: true`を使用してストリーミングレスポンスを受信する。**

### 理由

1. **タイムアウト回避**: ストリーミングを使用すると、データが継続的に流れるため、リバースプロキシの接続タイムアウトを回避できる
2. **ユーザーエクスペリエンス**: 応答が逐次表示され、処理が進行していることをユーザーに示せる
3. **業界標準**: OpenAI互換APIでは、ストリーミングは広くサポートされている標準機能

### 実装詳細

- `stream: true`をリクエストに含める
- レスポンスはServer-Sent Events（SSE）形式で返される
- 各チャンクを逐次パースし、`choices[0].delta.content`を蓄積する
- `done_reason`が`stop`または`length`になったらストリームを終了

## 影響

### ポジティブ

- 524タイムアウトエラーの発生を大幅に減少させる
- ユーザーに処理進行状況をリアルタイムで表示可能
- 大規模thinkingモデルの使用が可能になる

### ネガティブ

- クライアント側の実装が複雑化（SSEパース処理が必要）
- `stream: false`に比べて、エラーハンドリングが難しくなる（HTTPステータスコードは200で返り、エラーは途中のチャンクに含まれる場合がある）
- `num_predict`制限がない場合、モデルが無限に生成し続けるリスク（`stream: false`でも同様だが、ストリーミングでは検出が遅れる可能性）

## 関連する決定

- **ADR 002（予定）**: OllamaネイティブAPI（`/api/chat`）では、ストリーミングを使用すべきか？
  - 現在の検討事項: OllamaネイティブAPIでも同様の524エラーが発生しており、ストリーミング対応を検討中
  - ただし、OllamaのネイティブストリーミングはOpenAI互換とフォーマットが異なるため、別途実装が必要

## 備考

- 2024年4月時点で、Cloudflare WorkersのタイムアウトはFreeプランで10秒、Pro/Business/Enterpriseで300秒だが、**Cloudflareを経由する通常のHTTPリクエストは100秒制限**が適用される
- 524エラーは「A Timeout Occurred」であり、サーバー側ではなく、リバースプロキシ（Cloudflare）側で発生している
