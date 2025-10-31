EMAIL_SUMMARY_PROMPT = """You are an expert at analyzing and summarizing emails.

Analyze the following email and provide a concise, informative summary.


**Email**
{body}

**Entities Extracted:**
{entities}

**Instructions:**
1. Provide a 2 to 4 sentence summary of the email's main purpose and content and format it in a user friendly was that can be easily read (the less the better)
2. Highlight any key action items, decisions, or deadlines
3. Identify important people, projects, or topics mentioned
4. Note any attachments and their relevance

**Format your response with the following format:**
{{"Summary": [Your summary here]
"Key Points":
- [Point 1]
- [Point 2]
- [Point 3]
"Priority": [High/Medium/Low based on urgency and importance]}}
"""


REPORT_SUMMARY_PROMPT = """You are an expert at analyzing and summarizing business/technical reports.

Analyze the following report and provide a concise, informative summary.

**Report Content:**
{content}

**Report entities:**
{entities}

**Instructions:**
1. Provide a 3-4 sentence summary of the report's main purpose and findings
2. Identify the key objectives or goals
3. Highlight the main recommendations or conclusions
4. Note any data, metrics, or evidence presented
5. Assess the significance or impact of the report

**Format your response with the following format:**
{{"Summary": [Your summary here]
"Objectives": [Main goals or purpose]
"Key Findings":
- [Finding 1]
- [Finding 2]
- [Finding 3]
"Recommendations":
- [Recommendation 1]
- [Recommendation 2]
"Impact": [Expected impact or importance]}}

"""

SCIENTIFIC_PAPER_SUMMARY_PROMPT = """You are an expert at analyzing and summarizing scientific papers.

Analyze the following scientific paper and provide a clear, informative summary.

**Paper Content:**
{content}


**Entities:**
{entities}


**Instructions:**
1. Provide a 4-5 sentence summary in plain language
2. Identify the main research question or hypothesis or topic
3. Highlight the key methodology used
4. State the main findings or conclusions
5. Note the significance or implications of the work

****Format your response with the following format:****
{{"Summary": [Your summary here]
"Research Question": [Main question addressed]
"Methodology": [Brief description of approach]
"Key Findings":
- [Finding 1]
- [Finding 2]
- [Finding 3]
"Significance": [Why this matters]}}

"""