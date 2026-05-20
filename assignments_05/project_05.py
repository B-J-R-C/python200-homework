# ==========================================
# Week 5 Mini-Project: project_05.py
# Job Application Helper
# ==========================================

import json
from dotenv import load_dotenv
from openai import OpenAI

# Task 1: Setup and System Prompt
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# Define the System Prompt
YOUR_SYSTEM_PROMPT = """
You are a highly skilled, supportive job application coach specializing in helping career changers pivot into the tech industry. 
Your goal is to help users translate their past experience into compelling, results-oriented language.

Constraints:
1. Stay strictly focused on job application materials (resumes, cover letters, interview prep). If the user asks about unrelated topics, politely steer them back to their career transition.
2. Always remind the user to review, edit, and fact-check your output before submitting it to an employer.
3. Explicitly acknowledge that you may not know their specific target industry's exact norms, and encourage them to use their own judgment.
"""

"""
TASK 1 COMMENT: Why this system prompt?
I deliberately chose to narrow the persona to focus specifically on "pivoting into the tech industry." 
By explicitly defining the audience as career changers entering tech, the model will naturally favor 
language and skills highly valued in that space (like systems thinking, data analysis, or cross-functional 
collaboration) rather than giving generalized, one-size-fits-all corporate advice. I also included strict 
behavioral constraints to ensure the bot acts responsibly and manages the user's expectations.
"""


# ==========================================
# Task 2: Bullet Point Rewriter
# ==========================================
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.
    
    Return ONLY a valid JSON list of objects. Do not include markdown formatting like ```json.
    Each item should have two keys: "original" (the original bullet) and "improved" (your rewritten version).
    
    Bullet points:
    ```
    {bullet_text}
    ```
    """
    
    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages, temperature=0.5)
    
    try:
        parsed_bullets = json.loads(raw_response)
        print("\n--- Bullet Point Revisions ---")
        for idx, item in enumerate(parsed_bullets):
            print(f"\nBullet {idx + 1}:")
            print(f"Original: {item.get('original')}")
            print(f"Improved: {item.get('improved')}")
        return parsed_bullets
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        print("Raw response:", raw_response)
        return []

# Testing Task 2
print("--- Testing Bullet Rewriter ---")
test_bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(test_bullets)

"""
TASK 2 COMMENT: What makes these bullets weak, and what did the model suggest?
The original bullets weak because they are vague, passive, and lack measurable impact. 
Words like "helped," "made," and "worked" don't convey scale or success. The model's improvements 
typically inject stronger verbs (e.g., "resolved," "generated," "collaborated") and imply a focus 
on business value (e.g., improving satisfaction, delivering actionable insights, hitting deadlines).
"""


# ==========================================
# Task 3: Cover Letter Generator
# ==========================================
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    
    Here are two examples of the style and tone you should match:
    
    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.
    
    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.
    
    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    
    Opening:
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages, temperature=0.7)
    return response

# Testing Task 3
print("\n--- Testing Cover Letter Generator ---")
test_job = "Junior Data Engineer"
test_bg = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."
print(generate_cover_letter(test_job, test_bg))

"""
TASK 3 COMMENT: Why did you choose those examples and what does few-shot control?
I chose those examples because they clearly demonstrate the specific narrative arc of a successful 
career pivot: highlighting a past frustration or unique viewpoint, explaining the technical upskilling, 
and directly tying those two things to the target company. The few-shot pattern strictly controls the 
structural format and forces the model to mimic that confident, "show-don't-tell" tone, preventing 
it from reverting to generic corporate clichés like "I am a highly motivated individual."
"""


# ==========================================
# Task 4: Moderation Check
# ==========================================
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    
    if flagged:
        print("\n[System]: I cannot process that request. Please rephrase your input to adhere to safety guidelines.\n")
        return False
    return True

# Testing Task 4
print("\n--- Testing Moderation Endpoint ---")
safe_text = "I am applying for a job as a Python developer."
flagged_text = "I want to hack into my old company's database and steal customer passwords to get revenge."

print(f"Testing safe text: '{safe_text}' -> Safe? {is_safe(safe_text)}")
print(f"Testing flagged text: '{flagged_text}' -> Safe? {is_safe(flagged_text)}")


# ==========================================
# Task 5: The Chatbot Loop
# ==========================================
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]
    
    print("\n" + "=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
            
        # 3. Skip empty 
        if not user_input:
            continue
            
        # 4. Run moderation check 1st
        if not is_safe(user_input):
            continue  
            
        # 5. Check if  wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            
            if raw_bullets:
                rewrite_bullets(raw_bullets)
            else:
                print("Job Application Helper: You didn't enter any bullets! Let's continue.")
                
        # 6. Check if the user wants cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            
            # Moderation check on the sub-inputs
            if not is_safe(job_title) or not is_safe(background):
                continue
                
            draft = generate_cover_letter(job_title, background)
            print("\n--- Drafted Cover Letter Opening ---")
            print(draft)
            print("------------------------------------\n")
            
        # 7. Otherwise, handl as regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            
            messages.append({"role": "assistant", "content": reply})


# ==========================================
# Task 6: Ethics Reflection
# Option A - Comment Block
# ==========================================
"""
--- ETHICS REFLECTION (Option A) ---
1. How might the bot produce biased advice? 
Because large language models are trained on the internet, they heavily index on communication styles common in Western corporate culture. This means the tool might aggressively rewrite perfectly valid bullets to sound overly assertive or aggressive, punishing users whose cultural backgrounds favor more humble, collaborative, or indirect ways of describing their professional achievements. It essentially enforces a narrow, standardized idea of "professionalism."

2. What is one guardrail you would add if deploying professionally?
If deploying this professionally, I would add a strict "Fact-Checking Consent UI." Before allowing the user to copy the output text, the application would force them to check a box confirming that the generated bullet points or cover letters do not exaggerate their skills or invent false metrics. This structural guardrail forces a moment of human friction and accountability, reducing the likelihood of users blindly submitting hallucinated credentials.
"""

if __name__ == "__main__":
    # The tests run automatically when executing the script. 
    # The chatbot loop follows right after!
    run_chatbot()