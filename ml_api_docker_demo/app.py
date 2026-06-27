import os
import subprocess
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("=" * 60)
print("LLM CODE GENERATION + EXECUTION DEMO")
print("=" * 60)

task = input("\nEnter the coding task:\n")

user_input = input("\nEnter input for the generated program:\n")

prompt = f"""
You are an expert Python programmer.

Generate ONLY executable Python code.
Rules:

1. DO NOT use markdown.
2. DO NOT use triple backticks.
3. DO NOT explain anything.
4. Use ONLY ONE input() statement.
5. If multiple inputs are required, use input().split().
6. Never ask:
   Enter first number
   Enter second number
7. Print only the final answer.

Example:

Input:
10 20

Output:
30

Task:

{task}
"""

print("\nCalling Groq LLM API to generate code...\n")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You generate clean and safe beginner-level Python programs."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.1,
    max_tokens=500
)

generated_code = response.choices[0].message.content.strip()

# Remove markdown accidentally produced by model
generated_code = generated_code.replace("```python", "").replace("```", "").strip()

print("=" * 60)
print("GENERATED CODE")
print("=" * 60)
print(generated_code)

with open("generated_program.py", "w", encoding="utf-8") as f:
    f.write(generated_code)

#Opens (or creates) a local file named generated_program.py in write mode ("w"). The with statement ensures the file closes properly after writing the generated_code string into it.

print("\nGenerated code saved as generated_program.py")

print("\nRunning generated code with your input...\n")

try:
    result = subprocess.run(
        ["python", "generated_program.py"],
        input=user_input,
        text=True,
        capture_output=True,
        timeout=10
    )

#subprocess.run(...): Opens a background terminal process to execute your code.

    print("=" * 60)
    print("PROGRAM OUTPUT")
    print("=" * 60)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(result.stderr)

except subprocess.TimeoutExpired:
    print("Program stopped because it took too long.")