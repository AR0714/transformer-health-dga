import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_chatbot import TransformerChatbot

EVAL_QUESTIONS = [
    ("Normal",  "Is the transformer okay?"),
    ("Normal",  "What is the voltage right now?"),
    ("Normal",  "Is the oil temperature too hot?"),
    ("PD",      "How much hydrogen gas is inside?"),
    ("D2",      "Should I be worried about this transformer?"),
    ("T3",      "What is wrong with this transformer?"),
    ("D1",      "Should I shut it down immediately?"),
    ("Normal",  "How many years does this transformer have left?"),
    ("T2",      "What does the DGA diagnosis mean in simple words?"),
    ("Normal",  "Tell me something I should know about my transformer."),
]

def run_evaluation():
    print("=" * 70)
    print(" TRANSFORMER CHATBOT - 10-QUESTION EVALUATION")
    print("=" * 70)

    for i, (scenario, question) in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/10] Scenario: {scenario}")
        print(f"Q: {question}")
        print("-" * 50)
        bot = TransformerChatbot(fault_scenario=scenario)
        answer = bot.chat(question)
        print(f"A: {answer}")

    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE - All 10 questions answered!")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
