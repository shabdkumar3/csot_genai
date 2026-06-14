# Week 2 Submission

## What I Built
a research agent that runs in the terminal (textual tui). you type a question, it searches the web with serper, reads pages with requests, and for academic stuff it hits the alphaxiv mcp server for papers. then it writes an answer with the sources it used. two panels - left is the chat, right shows every tool call live so you can watch it work.

## How The Agent Loop Works
one messages list (system + user + assistant + tool results). every round i send it to the model with the tool schemas attached. the model either calls a tool -> i run it, push the result back as a `tool` message, loop again -> or just answers, and i stop. capped at 8 steps so it can't spin forever. web_search and web_fetch are normal functions i wrote. the alphaxiv tools come from the server itself - i call `list_tools()` and pass whatever it gives me to the model, so i'm not hardcoding paper tool names.

---

## Decisions I Made

### async agent, thread worker for the tui
the llm and web calls block, so if they ran on the ui thread the screen would freeze. i run the whole question in a textual thread worker (asyncio.run inside), which keeps the ui alive, and inside that thread async lets me actually `await` the mcp calls. web tools i push through asyncio.to_thread so the event loop stays free for the mcp stream. getting this layering right took the longest.

### discover mcp tools at runtime
instead of hardcoding `discover_papers` / `get_paper_content`, i ask the server for its tool list and convert that to openai schemas. if alphaxiv changes its tools, my agent still works.

### truncate fetched pages to 6000 chars
full pages blow up the context and cost, and the useful part is usually near the top anyway.

### mcp is best-effort, not a hard dependency
if the bridge isn't authed or node isn't installed, it catches the error, prints "alphaxiv unavailable", and just runs web-only instead of crashing.

---

## What Surprised Me
the alphaxiv mcp was not plug-and-play like lesson 3 made it look. it's streamable-http with oauth 2.0, and you can't just point an sse client at it - direct connections are cors locked to their own origin. the actual way is to run `npx mcp-remote` as a bridge that does the oauth login once and then talks stdio to my python. spent a while figuring that out before the papers part worked at all.

---

## What I'd Change
streaming the final answer token by token (right now you wait for the whole reply). a real save_research_note tool so the model can write findings to a file across sessions. and caching fetched pages so the same url isn't downloaded twice in one session.
