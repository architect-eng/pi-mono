import type { AgentTool } from "@mariozechner/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage, fauxToolCall } from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createHarness, type Harness } from "../harness.js";

/**
 * The faux provider always recomputes usage from the serialized context using
 * ceil(chars/4). We control context size by making tool results large enough
 * so the faux provider's estimated token count crosses the configured threshold.
 */

/** Generate a string of approximately `tokens` estimated tokens (chars/4 heuristic). */
function padToTokens(tokens: number): string {
	return "x".repeat(tokens * 4);
}

describe("issue #2871: mid-turn context window enforcement", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("aborts LLM call when estimated context exceeds threshold during tool loop", async () => {
		vi.useFakeTimers();

		// Context window 2000, reserve 400 -> threshold at 1600
		// keepRecentTokens: 200 so compaction keeps very little
		const contextWindow = 2000;
		const reserveTokens = 400;

		// Tool that returns a large result to push context past threshold
		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				// ~1200 tokens of tool result content
				content: [{ type: "text", text: padToTokens(1200) }],
				details: {},
			}),
		};

		const compactionSummaries: string[] = [];
		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: true,
					reserveTokens,
					keepRecentTokens: 200,
				},
			},
			tools: [bigTool],
			extensionFactories: [
				(pi) => {
					pi.on("session_before_compact", async (event) => {
						const summary = "compacted session";
						compactionSummaries.push(summary);
						return {
							compaction: {
								summary,
								firstKeptEntryId: event.preparation.firstKeptEntryId,
								tokensBefore: event.preparation.tokensBefore,
								details: {},
							},
						};
					});
				},
			],
		});
		harnesses.push(harness);

		// Flow:
		// 1st LLM call: context is small (user message). Returns tool call for big_read.
		// Tool executes -> 1200 tokens of result added to context.
		// 2nd LLM call attempt: beforeLlmCall estimates context > 1600 threshold -> ABORT.
		// Overflow recovery: strip error, compact, retry (via setTimeout).
		// 3rd LLM call (retry): compacted context -> small -> succeeds.

		let callCount = 0;
		harness.setResponses([
			() => {
				callCount++;
				return fauxAssistantMessage(fauxToolCall("big_read", {}), { stopReason: "toolUse" });
			},
			() => {
				callCount++;
				return fauxAssistantMessage("done after compaction");
			},
		]);

		// Start prompt - will resolve when the agent loop exits with the synthetic error.
		// Compaction + retry happen asynchronously after that.
		const promptPromise = harness.session.prompt("do something");
		await vi.advanceTimersByTimeAsync(0); // let the prompt start
		await promptPromise;

		// Compaction runs in the agent_end handler. The retry uses setTimeout(100).
		// Advance timers to trigger the retry.
		await vi.advanceTimersByTimeAsync(200);

		// Wait for the retry to complete
		await harness.session.agent.waitForIdle();

		// Verify compaction was triggered by the beforeLlmCall hook
		expect(compactionSummaries.length).toBeGreaterThanOrEqual(1);

		// Verify both faux responses were consumed (original + retry)
		expect(callCount).toBe(2);

		// Verify the session has a compaction entry
		const entries = harness.sessionManager.getEntries();
		const compactionEntries = entries.filter((e) => e.type === "compaction");
		expect(compactionEntries.length).toBeGreaterThanOrEqual(1);
	});

	it("does not interfere when context is within threshold", async () => {
		const contextWindow = 200_000;
		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: true,
					reserveTokens: 16384,
					keepRecentTokens: 20000,
				},
			},
		});
		harnesses.push(harness);

		harness.setResponses([fauxAssistantMessage("hello")]);

		await harness.session.prompt("hi");

		// Should complete normally without any compaction events
		const compactionEvents = harness.eventsOfType("compaction_start");
		expect(compactionEvents).toHaveLength(0);
	});

	it("does not abort on first LLM call when there is no prior usage", async () => {
		const contextWindow = 200_000;
		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: true,
					reserveTokens: 16384,
					keepRecentTokens: 20000,
				},
			},
		});
		harnesses.push(harness);

		harness.setResponses([fauxAssistantMessage("first response")]);

		await harness.session.prompt("hi");

		const assistantMessages = harness.session.messages.filter((m) => m.role === "assistant");
		expect(assistantMessages.length).toBeGreaterThan(0);
		const lastAssistant = assistantMessages[assistantMessages.length - 1] as AssistantMessage;
		expect(lastAssistant.stopReason).toBe("stop");
	});

	it("skips check when compaction is disabled", async () => {
		const contextWindow = 2000;

		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				content: [{ type: "text", text: padToTokens(1200) }],
				details: {},
			}),
		};

		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: false,
					reserveTokens: 400,
					keepRecentTokens: 200,
				},
			},
			tools: [bigTool],
		});
		harnesses.push(harness);

		let callCount = 0;
		harness.setResponses([
			() => {
				callCount++;
				return fauxAssistantMessage(fauxToolCall("big_read", {}), { stopReason: "toolUse" });
			},
			() => {
				callCount++;
				return fauxAssistantMessage("done");
			},
		]);

		await harness.session.prompt("do something");

		// Both LLM calls should have gone through since compaction is disabled
		expect(callCount).toBe(2);
	});

	it("does not falsely trigger on stale pre-compaction usage after compaction", async () => {
		vi.useFakeTimers();

		const contextWindow = 2000;

		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				content: [{ type: "text", text: padToTokens(1200) }],
				details: {},
			}),
		};

		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: true,
					reserveTokens: 400,
					keepRecentTokens: 200,
				},
			},
			tools: [bigTool],
			extensionFactories: [
				(pi) => {
					pi.on("session_before_compact", async (event) => ({
						compaction: {
							summary: "compacted",
							firstKeptEntryId: event.preparation.firstKeptEntryId,
							tokensBefore: event.preparation.tokensBefore,
							details: {},
						},
					}));
				},
			],
		});
		harnesses.push(harness);

		harness.setResponses([
			// First prompt: tool call -> large result -> triggers mid-turn compaction
			fauxAssistantMessage(fauxToolCall("big_read", {}), { stopReason: "toolUse" }),
			// After compaction + retry
			fauxAssistantMessage("first done"),
			// Second prompt: should work normally despite stale kept usage
			fauxAssistantMessage("second done"),
		]);

		// First prompt triggers compaction via beforeLlmCall
		const p1 = harness.session.prompt("first prompt");
		await vi.advanceTimersByTimeAsync(0);
		await p1;
		await vi.advanceTimersByTimeAsync(200);
		await harness.session.agent.waitForIdle();

		// Second prompt should NOT be blocked by stale kept usage
		const p2 = harness.session.prompt("second prompt");
		await vi.advanceTimersByTimeAsync(0);
		await p2;
		await vi.advanceTimersByTimeAsync(200);
		await harness.session.agent.waitForIdle();

		// Verify second prompt succeeded (not blocked by stale usage)
		const assistantMessages = harness.session.messages.filter((m) => m.role === "assistant") as AssistantMessage[];
		const lastAssistant = assistantMessages[assistantMessages.length - 1];
		expect(lastAssistant.stopReason).not.toBe("error");
	});
});
