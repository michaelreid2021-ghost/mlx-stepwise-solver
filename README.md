# **mlx-stepwise-solver**

A lightweight, local AI agent framework for structured problem-solving using MLX. This single-file Python script implements a stepwise solver that decomposes queries into plans, executes them with reasoning, and supports sub-pipelines for complex tasks. It leverages the Harmony Chat Format for prompt structuring, enabling compatibility with specific fine-tuned models.

## **Overview**

This solver operates in two main phases:

1. **Structured Planning**: Decomposes the user's query into a step-by-step plan with atomic tasks, types (e.g., single, iteration), and recommended tools.  
2. **Socratic Execution Loop**: Iteratively solves each step, showing reasoning in an "analysis" channel, and uses tools to store results. For intricate steps, it delegates to a sub-pipeline (plan → distill → solve → return).

The framework is designed for local inference on Apple silicon via MLX, emphasizing transparency, error handling, and modular tool integration. It's particularly suited for tasks requiring verifiable, multi-step reasoning, such as data analysis, decision-making, or automated workflows.

## **Motivation**

The core idea stems from experiments in guiding reasoning-trained models to "slow down" and form explicit plans. When direct planning prompts fell short, the approach shifted to a more organic flow: let the model freely brainstorm and "do its thing" on how to tackle the problem. This captures natural, emergent step-by-step insights, like a "steps answer popping out because they couldn't help it." From there, the model's own thoughts are distilled into a refined plan, "hooking" it into preferred inference pathways by reusing its self-expressed phrasing. (At least, that's the mental model I had in mind trying to leverage the model's tendencies for self-reinforcement.).


## **Features**

* **JSON-Based State Management**: Tracks plan execution with statuses, answers, and proofs.  
* **Tool Support**: Built-in tools include:  
  * scratchpad: Temporary content storage.  
  * save\_variable: Persistent variable saving with proof.  
  * memory\_buffer: Step summaries and results aggregation.  
  * solve\_current\_step: Delegates to a full sub-pipeline for nested reasoning.  
* **Harmony Chat Format Integration**: Uses special tokens (e.g., \<|start|\>, \<|message|\>, \<|channel|\>) for roles (system, developer, user, assistant), channels (analysis, commentary, final), and tool calls. This ensures structured outputs and multi-turn compatibility.  
* **Sub-Pipeline Delegation**: For complex steps, runs a mini-loop with planning, distillation, solving, and synthesis.  
* **Customizable Reasoning Levels**: Supports low/medium/high reasoning via command-line args.  
* **Error Handling**: Robust parsing for JSON outputs and tool calls.

## **Compatibility**

This script is specifically designed for models trained using the **Harmony Chat Format**, which structures conversations with roles, channels, and special tokens to support reasoning, tool calls, and multi-turn interactions. Tested with:

* openai/gpt-oss20b  
* openai/gpt-oss120b

It has not been tested with other models using similar chat structures. For details on the format, see the [OpenAI Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony), which includes syntax for tokens like \<|start|\>, \<|message|\>, \<|channel|\>, tool namespaces, and structured outputs.

The script uses MLX for efficient local inference on compatible hardware (e.g., Apple M-series chips). Ensure your model is converted to MLX format using mlx\_lm.convert.

## **Installation**

1. Install dependencies:  
   text  
   pip install mlx mlx-lm

Clone the repo:  
text  
git clone https://github.com/yourusername/mlx-stepwise-solver.git

2. cd mlx-stepwise-solver

## **Usage**

Run the script with a model path and prompt:

text  
python main.py \--model /path/to/mlx-model \--prompt "Solve: What is the capital of Japan?"

### **Command-Line Arguments**

* \--model (required): Path to the MLX-converted model directory.  
* \--prompt (required): The user's initial question or task.  
* \--reasoning (optional): Reasoning level (low, medium, high; default: medium).  
* \--max-tokens (optional): Max tokens for generation (default: 16000).  
* \--temp (optional): Sampling temperature (default: 0.7).  
* \--top-p (optional): Top-p sampling (default: 0.95).  
* \--sub-max-tokens (optional): Token limit for sub-pipelines (default: 4000).

### **Example Output**

For a simple prompt like \--prompt "What is 2 \+ 2?", the script will:

* Generate a plan (e.g., single arithmetic step).  
* Execute with analysis.  
* Output a final answer like: "4"

For complex prompts, it delegates sub-tasks and aggregates proofs.

## **How It Works**

1. **Planning Phase**: Uses a tool-less system prompt to generate a JSON plan array with steps, tasks, types, and recommended tools.  
2. **Execution Phase**: Loops through steps, prompting the model to reason and call tools. Parses responses for analysis text and tool calls.  
3. **Sub-Pipeline**: When solve\_current\_step is called, runs a nested loop to plan, distill, solve, and return a verified answer.  
4. **Final Synthesis**: Compiles completed steps into a human-readable summary.

## **Limitations**

* Single-file implementation; extend by adding custom tools to the functions namespace.  
* Relies on model adherence to Harmony format for parsing (e.g., via extract\_json\_from\_response).  
* No built-in persistence beyond in-memory state.  
* Tested only on specified models; may require prompt tweaks for others.

## **Contributing**

Contributions welcome\! Fork the repo and submit pull requests for bug fixes, new tools, or MLX optimizations.

## **License**

MIT License. Copyright (c) 2025 Michael S. Reid.

## **References**

* [OpenAI Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony) – Guide to the chat format used in this script.

