import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createHarness, type Harness } from "../harness.ts";

/**
 * The faux provider always recomputes usage from the serialized context using
 * ceil(chars/4). We control context size by making tool results large enough
 * so the faux provider's estimated token count crosses the configured threshold.
 */

/** Generate a string of approximately `tokens` estimated tokens (chars/4 heuristic). */
function padToTokens(tokens: number): string {
	return "x".repeat(tokens * 4);
}

describe("issue #5512: mid-turn context window enforcement", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("stops after a tool turn, compacts, and resumes when estimated context exceeds threshold", async () => {
		// Context window 1200, reserve 100 -> threshold at 1100
		// keepRecentTokens keeps the current large tool turn but leaves older history to summarize.
		const contextWindow = 1200;
		const reserveTokens = 100;

		// Tool that returns a large result to push context past threshold
		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				// ~1200 tokens of tool result content
				content: [{ type: "text", text: padToTokens(800) }],
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
					keepRecentTokens: 805,
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
		// Tool executes -> large result added to context.
		// shouldStopAfterTurn estimates context > 1100 threshold and exits cleanly.
		// AgentSession compacts, then resumes the tool loop with agent.continue().
		// 2nd LLM call: compacted context -> small -> succeeds.

		let callCount = 0;
		harness.setResponses([
			fauxAssistantMessage("seed turn"),
			() => {
				callCount++;
				return fauxAssistantMessage(fauxToolCall("big_read", {}), { stopReason: "toolUse" });
			},
			() => {
				callCount++;
				return fauxAssistantMessage("done after compaction");
			},
		]);

		// Seed a prior turn so there is history for compaction to summarize.
		await harness.session.prompt("seed");
		await harness.session.prompt("do something");

		// Verify compaction was triggered by the post-turn stop hook.
		expect(compactionSummaries.length).toBeGreaterThanOrEqual(1);
		expect(harness.eventsOfType("compaction_start").map((event) => event.reason)).toContain("threshold");

		// Verify both faux responses were consumed (original + resumed tool loop).
		expect(callCount).toBe(2);
		const assistantMessages = harness.session.messages.filter((m) => m.role === "assistant") as AssistantMessage[];
		expect(assistantMessages.every((message) => message.stopReason !== "error")).toBe(true);

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

	it("does not stop before the first LLM call when there is no prior usage", async () => {
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
		const contextWindow = 1200;

		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				content: [{ type: "text", text: padToTokens(800) }],
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

		// Both LLM calls should run since compaction is disabled
		expect(callCount).toBe(2);
	});

	it("does not falsely trigger on stale pre-compaction usage after compaction", async () => {
		const contextWindow = 1200;

		const bigTool: AgentTool = {
			name: "big_read",
			label: "Big Read",
			description: "Returns a large result",
			parameters: Type.Object({}),
			execute: async () => ({
				content: [{ type: "text", text: padToTokens(800) }],
				details: {},
			}),
		};

		const harness = await createHarness({
			models: [{ id: "test-model", contextWindow }],
			settings: {
				compaction: {
					enabled: true,
					reserveTokens: 100,
					keepRecentTokens: 805,
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
			fauxAssistantMessage("seed turn"),
			// First prompt: tool call -> large result -> triggers post-turn compaction
			fauxAssistantMessage(fauxToolCall("big_read", {}), { stopReason: "toolUse" }),
			// After compaction + retry
			fauxAssistantMessage("first done"),
			// Second prompt: should work normally despite stale kept usage
			fauxAssistantMessage("second done"),
		]);

		// Seed a prior turn so there is history for compaction to summarize.
		await harness.session.prompt("seed");

		// First prompt triggers compaction via shouldStopAfterTurn.
		await harness.session.prompt("first prompt");

		// Second prompt should NOT be blocked by stale kept usage.
		await harness.session.prompt("second prompt");

		// Verify second prompt succeeded (not blocked by stale usage)
		const assistantMessages = harness.session.messages.filter((m) => m.role === "assistant") as AssistantMessage[];
		const lastAssistant = assistantMessages[assistantMessages.length - 1];
		expect(lastAssistant.stopReason).not.toBe("error");
	});
});
