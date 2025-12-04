#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created By: Michael S. Reid
# Date: 09-25-2025
#mlx_stepwise_agent.py
# last update: Adds solve_current_step tool that delegates a full sub-pipeline (plan→distill→solve→return).

import argparse
import mlx.core as mx  # noqa: F401 (import retained for compatibility with MLX envs)
import mlx_lm
from mlx_lm.sample_utils import make_sampler
import re
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple
import pprint
import traceback

# --- UTILITY FUNCTIONS (Unchanged or Extended) ---
def parse_tool_calls(response: str) -> List[Tuple[str, str]]:
    calls = []
    pattern = re.compile(
        r"to=functions\.([a-zA-Z_]+)\s*"
        r"(?:<\|constrain\|>[^<]*)?"
        r"<\|message\|>(.*?)<\|call\|>",
        re.DOTALL
    )
    for name, payload in pattern.findall(response):
        calls.append((name, payload.strip()))
    return calls

def extract_json_from_response(response: str) -> Dict:
    """
    Extracts the LAST JSON object that immediately follows '<|message|>{'
    and returns it as a Python dict. Raises ValueError on failure.
    """
    anchor = '<|message|>{'
    start_index = response.rfind(anchor)
    if start_index == -1:
        raise ValueError("Could not find a valid JSON anchor in the model's response.")

    json_start_index = start_index + len(anchor) - 1
    text_from_start = response[json_start_index:]

    brace_count = 0
    end_index = -1
    for i, char in enumerate(text_from_start):
        if char == '{': brace_count += 1
        elif char == '}': brace_count -= 1
        if brace_count == 0 and i > 0:
            end_index = i + 1
            break

    if end_index == -1:
        raise ValueError("Could not find the matching closing brace '}' for the JSON object.")

    json_str = text_from_start[:end_index]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse extracted JSON string: {e}\n--- Extracted String ---\n{json_str}\n------------------------")

def extract_analysis_text(response: str) -> str:
    match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>)", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def _chunk_text(part) -> str:
    if isinstance(part, dict):
        return part.get("text", "")
    text = getattr(part, "text", None)
    return text if isinstance(text, str) else ""

def generate_until_stop(model, tokenizer, prompt: str, sampler, max_tokens: int, stop_markers: List[str]):
    buffer = ""
    print("\n--- Model Generation ---")
    for part in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=max_tokens):
        chunk = _chunk_text(part)
        if not chunk:
            continue

        temp_buffer = buffer + chunk
        found_stop = None
        stop_position = -1

        for marker in stop_markers:
            pos = temp_buffer.find(marker)
            if pos != -1:
                if found_stop is None or pos < stop_position:
                    found_stop = marker
                    stop_position = pos

        if found_stop:
            print(temp_buffer[:stop_position], end="", flush=True)
            buffer = temp_buffer[:stop_position + len(found_stop)]
            break
        else:
            print(chunk, end="", flush=True)
            buffer += chunk

    print("\n--- End Generation ---")
    return buffer

def create_final_summary(state: Dict) -> str:
    """Creates a clean, human-readable summary of the completed plan."""
    summary_lines = [f"Original Question: {state['original_question']}\n", "Summary of Verified Steps:"]
    for step in state.get("plan", []):
        if step.get("status") != "completed":
            continue
        step_num = step.get('step')
        task = step.get('task')
        answer = step.get('step_answer')
        summary_lines.append(f"  - Step {step_num}: {task} -> Result: {answer}")
    return "\n".join(summary_lines)

# --- NEW: Sub-pipeline runner for solve_current_step ---
def run_step_subpipeline(
    model,
    tokenizer,
    sampler,
    base_system_prompt: str,
    goal: str,
    must_return: str,
    context: Any = None,
    constraints: List[str] = None,
    quality_bar: str = None,
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Runs a local micro-pipeline: plan -> distill -> solve -> return.
    Returns: {"answer": ..., "proof": str, "artifacts": {...}, "plan_used": {"draft":[...], "distilled":[...]}}
    """
    constraints = constraints or []

    # ---------- 1) PLAN ----------
    planning_dev_prompt = (
        "You are a Master Planner. Produce a short, bounded plan (2-6 steps) to achieve the goal. "
        "Each step must be atomic and verifiable. Output JSON starting with '{' and use key 'plan' as an array."
    )
    planning_prompt = (
        f"<|start|>system<|message|>{base_system_prompt}<|end|>"
        f"<|start|>developer<|message|>{planning_dev_prompt}<|end|>"
        f"<|start|>user<|message|>Goal: {goal}\nMust return: {must_return}\n"
        f"Constraints: {constraints}\n"
        f"Context: {json.dumps(context, ensure_ascii=False) if context is not None else 'null'}<|end|>"
        f"<|start|>assistant<|message|>"
    )
    draft_plan_raw = mlx_lm.generate(model, tokenizer, prompt=planning_prompt, max_tokens=max_tokens, sampler=sampler)
    draft = extract_json_from_response(draft_plan_raw)

    # ---------- 2) DISTILL ----------
    distill_dev_prompt = (
        "Tighten the plan: enforce atomicity, add acceptance_criteria and stop_criteria. "
        "Keep 2-6 steps. Output JSON starting with '{' including keys: plan[], acceptance_criteria, stop_criteria."
    )
    distill_prompt = (
        f"<|start|>system<|message|>{base_system_prompt}<|end|>"
        f"<|start|>developer<|message|>{distill_dev_prompt}<|end|>"
        f"<|start|>user<|message|>Draft plan:\n```json\n{json.dumps(draft, ensure_ascii=False)}\n```\n"
        f"Quality bar: {quality_bar or 'be precise'}<|end|>"
        f"<|start|>assistant<|message|>"
    )
    distilled_raw = mlx_lm.generate(model, tokenizer, prompt=distill_prompt, max_tokens=max_tokens, sampler=sampler)
    distilled = extract_json_from_response(distilled_raw)
    distilled_plan = distilled.get("plan", [])
    if not isinstance(distilled_plan, list) or not (2 <= len(distilled_plan) <= 6):
        raise RuntimeError("Distilled plan invalid or out of allowed bounds (2-6 steps).")

    # ---------- 3) SOLVE ----------
    # Reuse the same tool-aware environment as the main execution.
    # Explicitly declare the available tools in the developer prompt.
    sub_exec_dev_prompt = (
        "# Instructions\n"
        "Execute only the current micro-step. Show your work in 'analysis'. "
        "When ready, call exactly one tool to record the micro-result. "
        "Continue until all micro-steps complete, then synthesize the final answer matching 'must_return'.\n\n"
        "# Tools\n"
        "## functions\n"
        "namespace functions {\n"
        "  type scratchpad = (_: { content: string | number }) => any;\n"
        "  type save_variable = (_: { variable_name: string, value: any, proof: string }) => any;\n"
        "  type memory_buffer = (_: { step_summary: string, result: any, last_step: bool }) => any;\n"
        "}\n"
        "<|end|>\n"
    )
    sub_conversation = (
        f"<|start|>system<|message|>{base_system_prompt}<|end|>"
        f"<|start|>developer<|message|>{sub_exec_dev_prompt}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>Beginning sub-execution for the current step.<|end|>"
    )

    artifacts: Dict[str, Any] = {}
    proof_chunks: List[str] = []

    for i, micro in enumerate(distilled_plan, 1):
        task = micro.get("task", "").strip()
        if not task:
            raise RuntimeError(f"Micro-step {i} missing 'task'.")

        exec_payload = {
            "goal": goal,
            "must_return": must_return,
            "step": i,
            "task": task,
            "context": context,
            "constraints": constraints
        }
        sub_user = (
            f"Execute micro-step {i}: {task}\n\n"
            f"Context:\n```json\n{json.dumps(exec_payload, ensure_ascii=False)}\n```"
        )
        prompt = sub_conversation + f"<|start|>user<|message|>{sub_user}<|end|><|start|>assistant"
        micro_resp = generate_until_stop(model, tokenizer, prompt, sampler, max_tokens, ["<|call|>"])
        proof_chunks.append(extract_analysis_text(micro_resp))

        tcs = parse_tool_calls(micro_resp)
        if not tcs:
            raise RuntimeError(f"Micro-step {i} did not call a tool.")
        m_name, m_payload = tcs[0]
        try:
            m_obj = json.loads(m_payload) if m_payload else {}
        except json.JSONDecodeError:
            m_obj = {"value": m_payload}

        artifacts[f"{m_name}_{i}"] = m_obj

        # Acknowledge tool to keep transcript coherent
        sub_conversation += f"<|start|>assistant{micro_resp}"
        sub_conversation += (
            f"<|start|>functions.{m_name} to=assistant<|channel|>commentary"
            f"<|message|>{{'status':'OK','micro_step':{i}}}<|end|>"
        )

    # ---------- 4) SYNTHESIZE ANSWER ----------
    synth_user = (
        f"All micro-steps are executed. Provide the final answer that satisfies 'must_return'. "
        f"Then provide a compact proof (<200 words).\n\n"
        f"Acceptance criteria: {distilled.get('acceptance_criteria','')}\n"
        f"Stop criteria: {distilled.get('stop_criteria','')}\n"
        f"Artifacts:\n```json\n{json.dumps(artifacts, ensure_ascii=False)}\n```"
    )
    synth_prompt = sub_conversation + f"<|start|>user<|message|>{synth_user}<|end|><|start|>assistant<|channel|>final<|message|>"
    synth_raw = mlx_lm.generate(model, tokenizer, prompt=synth_prompt, max_tokens=max_tokens, sampler=sampler)
    final_text = synth_raw.split("<|message|>", 1)[-1]
    final_text = final_text.replace("<|return|>", "").replace("<|end|>", "").strip()

    return {
        "answer": final_text,
        "proof": "\n".join(pc for pc in proof_chunks if pc).strip(),
        "artifacts": artifacts,
        "plan_used": {"draft": draft, "distilled": distilled}
    }

def main():
    parser = argparse.ArgumentParser(description="Run the 'Mind' Solver with Structured Planning.")
    parser.add_argument("--model", required=True, help="Path to the MLX model.")
    parser.add_argument("--prompt", required=True, help="The initial user question.")
    parser.add_argument("--reasoning", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--sub-max-tokens", type=int, default=4000, help="Token cap for sub-pipeline phases.")
    args = parser.parse_args()

    print("[INFO] Loading MLX model and tokenizer...")
    model, tokenizer = mlx_lm.load(args.model)
    print(f"[INFO] Model loaded. Using reasoning level: {args.reasoning}")

    current_date = datetime.now().strftime("%Y-%m-%d")

    # --- ARCHITECTURAL PROMPTS ---

    # *** RESTORED: Your custom system prompt for the tool-less planning phase ***
    planning_system_prompt = f'''You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {current_date}
Reasoning: {args.reasoning}
# Valid channels: analysis, commentary, final. Channel must be included for every message.
'''

    # The main system prompt for the execution phase, which IS tool-aware
    execution_system_prompt = f'''{planning_system_prompt}
Calls to these tools must go to the commentary channel: 'functions'.
'''

    developer_prompt_planning =  '''# Instructions
You are a Master Planner. Your only task is to analyze the user's request and decompose it into a detailed, step-by-step plan.
# Response Formats
## structured_plan
// Use the responseSchema below to generate a valid plan.
{
"title": "A plan to solve the user's request.",
"type": "object",
"properties": {
"plan": {
"type": "array",
"description": "The ordered sequence of steps to solve the problem.",
"items": {
"type": "object",
"properties": {
"step": { "type": "integer", "description": "The step number, starting from 1." },
"task": { "type": "string", "description": "A clear, concise description of the task for this step." },
"task_type": { "type": "string", "enum": ["single", "iteration", "coalesce", "summarize"], "description": "The nature of the task." },
"step_answer": { "type": "any", "description": "a placeholder for where an answer will later be provided.." },
"proof": { "type": "string", "description": "a placeholder for where the agent solving it will prove their answer" },
"recommended_tool": { "type": "string", "enum": ["scratchpad", "save_variable", "memory_buffer"], "description": "The single most appropriate tool to call after this step is executed." }
},
"required": ["step", "task", "task_type", "recommended_tool"]
}
}
},
"required": ["plan"]
}
<|end|>
'''

    developer_prompt_execution = '''# Instructions
You are a helpful assistant to any users request
Your job is to focus *only* on the step marked with `"status": "current"`. Take your time show your work, be thorough.
Once you believe you are ready call the `recommended_tool` and store the information that completes the task with your result. 
Do not solve any other steps until you have written your findings using the appropriate function that is named functions.<tool named in the plan> after each step.
If the current step involves multi-step reasoning, recalling multiple facts and then combining portions of them etc, make use of a special planning agent we have made available to you. You can request that the agent solve a portion of the current step on your behalf by calling the tool   functions.solve_current_step simply  by clearly asking it to solve a problem, and wait for its response. 
# Tools
## functions
namespace functions {
  type scratchpad = (_: { content: string | number }) => any;
  type save_variable = (_: { variable_name: string, value: any, proof: string }) => any;
  type memory_buffer = (_: { step_summary: string, result: any, last_step: bool }) => any;
  type solve_current_step = (_: {
    goal: string,
    must_return: string,
    context?: any,
    constraints?: string[],
    quality_bar?: string
  }) => any;
}
<|end|>
'''
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    # =================================================================
    # PHASE 1: STRUCTURED PLANNING
    # =================================================================
    print(f"\n{'='*20} PHASE 1: STRUCTURED PLANNING {'='*20}", flush=True)

    # *** RESTORED: Using YOUR tool-less system prompt for the planning phase ***
    planning_prompt = (
        f"<|start|>system<|message|>{planning_system_prompt}<|end|>"
        f"<|start|>developer<|message|>{developer_prompt_planning}<|end|>"
        f"<|start|>user<|message|>{args.prompt}<|end|>"
        f"<|start|>assistant<|message|>"
    )

    print("[AGENT] Requesting structured plan from model...")
    raw_plan_response = mlx_lm.generate(model, tokenizer, prompt=planning_prompt, max_tokens=args.max_tokens, sampler=sampler)

    print("--- Full Plan Response from Model ---")
    print(raw_plan_response)
    print("------------------------------------")

    try:
        if '"plan":' not in raw_plan_response:
            raise ValueError("Model response did not contain a 'plan' array.")
        structured_plan = extract_json_from_response(raw_plan_response)
        if not structured_plan.get("plan"):
            raise ValueError("Parsed JSON is valid, but the 'plan' array is missing or empty.")
        print("[AGENT] Successfully parsed structured plan from model.")
    except ValueError as e:
        print(f"[FATAL] Could not parse plan from model's Phase 1 response: {e}")
        return

    # =================================================================
    # AGENT INTERVENTION
    # =================================================================
    print("\n[AGENT] Transforming plan into executable state machine...")
    execution_state: Dict[str, Any] = {"original_question": args.prompt, "plan": []}
    for i, step in enumerate(structured_plan.get("plan", [])):
        step["step_answer"] = None
        step["proof"] = None
        step["status"] = "current" if i == 0 else "pending"
        execution_state["plan"].append(step)
    print("[AGENT] Transformation complete. Starting Socratic Execution Loop.")
    pprint.pprint(execution_state)

    # =================================================================
    # PHASE 2: SOCRATIC EXECUTION LOOP
    # =================================================================
    # *** CORRECTED: Using the tool-aware system prompt for the execution phase ***
    conversation_history = (
        f"<|start|>system<|message|>{execution_system_prompt}<|end|>"
        f"<|start|>developer<|message|>{developer_prompt_execution}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>Okay, I have created a plan. I will now execute it step-by-step to ensure accuracy.<|end|>"
    )

    turn_counter = 0
    while True:
        turn_counter += 1
        print(f"\n{'='*20} EXECUTION TURN {turn_counter} {'='*20}", flush=True)

        current_step_idx = next((i for i, step in enumerate(execution_state["plan"]) if step["status"] == "current"), -1)

        if current_step_idx == -1:
            print("[AGENT] All steps completed. Proceeding to final answer.")
            break

        current_task_desc = execution_state["plan"][current_step_idx]["task"]
        execution_prompt_user_part = (
            f"Excellent. We have completed {turn_counter - 1} step(s). Let's proceed.\n\n"
            f"Our current task is **Step {current_step_idx + 1}: \"{current_task_desc}\"**.\n\n"
            f"Please only work on the current step and save your work before proceeding.\n"
            f"```json\n{json.dumps(execution_state, indent=2)}\n```"
        )

        prompt_for_model = conversation_history + f"<|start|>user<|message|>{execution_prompt_user_part}<|end|><|start|>assistant"

        # ==========================================================
        # PROMPT PREVIEW
        # ==========================================================
        print("\n" + "="*25 + f" PROMPT FOR TURN {turn_counter} " + "="*25)
        print(prompt_for_model)
        print("=" * (52 + len(str(turn_counter))))
        # ==========================================================
        response = generate_until_stop(model, tokenizer, prompt_for_model, sampler, args.max_tokens, ["<|call|>"])

        try:
            new_reasoning = extract_analysis_text(response)
            tool_calls = parse_tool_calls(response)
            if not tool_calls:
                print("[ERROR] Model failed to call a tool this turn. Halting.")
                break

            func_name, payload_str = tool_calls[0]
            print(f"[AGENT] Model called 'functions.{func_name}'.")

            # --- NEW: solve_current_step branch ---
            if func_name == "solve_current_step":
                try:
                    payload_obj = json.loads(payload_str) if payload_str else {}
                except json.JSONDecodeError:
                    payload_obj = {}

                goal = payload_obj.get("goal") or current_task_desc
                must_return = payload_obj.get("must_return", "plain text answer")

                subres = run_step_subpipeline(
                    model, tokenizer, sampler,
                    base_system_prompt=execution_system_prompt,
                    goal=goal,
                    must_return=must_return,
                    context=payload_obj.get("context"),
                    constraints=payload_obj.get("constraints"),
                    quality_bar=payload_obj.get("quality_bar"),
                    max_tokens=min(args.max_tokens, args["sub_max_tokens"] if isinstance(args, dict) and "sub_max_tokens" in args else args.sub_max_tokens)
                )

                # Record results into the current step
                execution_state["plan"][current_step_idx]["proof"] = subres.get("proof")
                execution_state["plan"][current_step_idx]["step_answer"] = subres.get("answer")
                execution_state["plan"][current_step_idx]["status"] = "completed"

                # Update transcript so the model "sees" the tool result
                conversation_history += f"<|start|>user<|message|>{execution_prompt_user_part}<|end|>"
                conversation_history += f"<|start|>assistant{response}"
                conversation_history += (
                    f"<|start|>functions.solve_current_step to=assistant<|channel|>commentary"
                    f"<|message|>{json.dumps({'status': 'OK', 'step_completed': current_step_idx + 1}, ensure_ascii=False)}<|end|>"
                )

                # Advance pointer
                if current_step_idx + 1 < len(execution_state["plan"]):
                    execution_state["plan"][current_step_idx + 1]["status"] = "current"
                    print(f"[AGENT] Step {current_step_idx + 1} complete via sub-pipeline. Next step is {current_step_idx + 2}.")
                else:
                    print("[AGENT] Final step of the plan is complete.")
                continue  # Next turn

            # --- Default path: other tools (scratchpad/save_variable/memory_buffer) ---
            try:
                payload_obj = json.loads(payload_str)
                if isinstance(payload_obj, dict) and len(payload_obj) == 1:
                    unboxed_answer = list(payload_obj.values())[0]
                else:
                    unboxed_answer = payload_obj
            except json.JSONDecodeError:
                unboxed_answer = payload_str

            execution_state["plan"][current_step_idx]["proof"] = new_reasoning
            execution_state["plan"][current_step_idx]["step_answer"] = unboxed_answer
            execution_state["plan"][current_step_idx]["status"] = "completed"

            conversation_history += f"<|start|>user<|message|>{execution_prompt_user_part}<|end|>"
            conversation_history += "<|start|>assistant" + response
            conversation_history += (
                f"<|start|>functions.{func_name} to=assistant<|channel|>commentary"
                f"<|message|>{{'status': 'OK', 'step_completed': {current_step_idx + 1}}}<|end|>"
            )

            if current_step_idx + 1 < len(execution_state["plan"]):
                execution_state["plan"][current_step_idx + 1]["status"] = "current"
                print(f"[AGENT] Step {current_step_idx + 1} complete. Next step is {current_step_idx + 2}.")
            else:
                print("[AGENT] Final step of the plan is complete.")
        except Exception as e:
            print(f"[ERROR] An error occurred during execution loop: {e}. Halting.")
            traceback.print_exc()
            break

    # =================================================================
    # FINAL ANSWER GENERATION
    # =================================================================
    print(f"\n{'='*20} FINAL ANSWER GENERATION {'='*20}", flush=True)

    execution_summary = create_final_summary(execution_state)

    final_user_prompt = (
        "Excellent work. All steps of our plan are now complete and verified. "
        "Based on the following summary of our findings, please provide a comprehensive "
        "final answer to my original question.\n\n"
        f"--- Execution Summary ---\n{execution_summary}\n-------------------------"
    )

    final_prompt = conversation_history + f"<|start|>user<|message|>{final_user_prompt}<|end|><|start|>assistant<|channel|>final<|message|>"
    print(f"final prompt: {final_prompt}")
    print("[AGENT] Requesting final answer from model...")
    final_response = mlx_lm.generate(model, tokenizer, prompt=final_prompt, max_tokens=args.max_tokens, sampler=sampler)
    print(f"final response: {final_response}")
    if "<|message|>" in final_response:
        clean_answer = final_response.split("<|message|>", 1)[-1].strip()
        for stop_token in ["<|return|>", "<|end|>"]:
            if clean_answer.endswith(stop_token):
                clean_answer = clean_answer[:-len(stop_token)].strip()
    else:
        clean_answer = final_response

    print("\n" + "="*50)
    print("Final Output:")
    print(clean_answer)
    print("="*50)

if __name__ == "__main__":
    main()
