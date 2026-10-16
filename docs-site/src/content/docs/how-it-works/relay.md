---
title: The relay
description: The request queue, request cleanup and tool-call repair that sit between agents and the model.
---

## Queue

The host runs at most as many requests at once as the model has slots (2 by default). The rest wait in one first-come queue: 16 waiting, 10 minutes at most.

When the queue is full, the caller gets an overload error in its own API format: Anthropic `overloaded_error` with HTTP 529, or an OpenAI-style 503 with `Retry-After`. A caller that disconnects leaves the queue, or its upstream request is closed.

## Request cleanup

Strict chat templates reject a second system message, unknown roles, or two user messages in a row, and agents send them. The relay reshapes requests for all three APIs (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses) so any model's template accepts them.

It also keeps the prompt prefix stable, for example by removing Claude Code's per-session billing-header line and token counter, so llama.cpp can reuse its cache.

To see exactly what changed, set `HEARTHWORK_RELAY_DUMP=<folder>`: every request is written before and after cleanup.

## Tool-call repair

Some models write a tool call as plain text in a format the chat template does not parse. When a request offered tools and the reply holds no structured call, the relay looks for one in the text and rewrites the reply into the API's own form (Anthropic `tool_use`, OpenAI `tool_calls`, Responses `function_call`), streaming or not.

Formats tried: Mistral `[TOOL_CALLS]`, `[Calling tool: f({...})]`, `<tool_call>{json}</tool_call>`, `<function=name>...`, XML-style tags, pythonic `f(a=1)` and raw JSON.

A call is accepted only when the tool exists in the request and its arguments fit the schema's required keys and basic types. Otherwise the reply is untouched. Streams are delayed only when the text starts to look like a call. Each repair is logged and shown by [`hearthwork doctor`](/troubleshooting/doctor/). Turn repair off with `HEARTHWORK_TOOL_REPAIR=0`.

## Context errors

A prompt that does not fit gets a clear error in the API's own format, with the command to raise the context. Requests are never truncated.
