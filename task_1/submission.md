# Week 1 Submission

## What I Built
terminal chatbot with multi-turn conversation, api key loaded from .env, and context compaction when tokens get too high. built using groq api and openai sdk.

---

## Decisions I Made

### Groq over OpenRouter
openrouter was giving 401 errors so i just switched to groq. both are openai-compatible so literally just changed the base_url and model name. groq is also faster so works better for a terminal chatbot.

### OpenAI SDK instead of raw requests
the track says no frameworks, not no libraries. openai sdk is just a wrapper over http — its literally doing the same requests.post under the hood. i know whats happening inside, the sdk just saves me from writing response.json()["choices"][0] every time.

### manual history list
llms are stateless, they remember nothing between calls. so i maintain a history list and send the whole thing every api call. thats literally the only way to make it feel like a conversation.

### compaction using total_tokens
response.usage.total_tokens already tells me how big the current request was (history + reply combined). so i just check that directly instead of counting manually. when it crosses 1000 i summarise.

### summarising only last 6 messages
no point sending the full history to summarise — thats wasteful and costs more tokens. the last 6 messages have the most relevant context anyway. older stuff either doesnt matter or was already summarised before.

### try-except with history.pop()
if the api call fails after i already appended the user message, history has a dangling message with no reply. next call would be messed up. so i pop it on failure.

---

## What I Learned
- llm apis are just http post requests, nothing fancy
- stateless nature of llms and why you have to send history every time
- tokens and why context window matters
- api key hygiene, .env and .gitignore
- difference between a framework (langchain) and a library (openai sdk)

---

## What I'd Change
would add streaming next time — i know how it works (stream=True, SSE parsing, flush=True) but kept this simple for now.