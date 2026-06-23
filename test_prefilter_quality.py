"""Honest A/B: deterministic pre-filter vs gpt-5-mini (the production path).
For each name: does the pre-filter handle it, and if so does its output match the LLM?
Outputs /tmp/prefilter_eval.json for an independent Claude judge to rate."""
import json, os, sys
sys.path.insert(0, "/tmp/acc")
from app.llm.postprocess import deterministic_prefilter
from openai import OpenAI

SYSTEM_PROMPT = open("/tmp/sys_prompt.txt").read().strip()
client = OpenAI()

# Representative GMaps-style sample across every category
NAMES = [
    # simple mechanical (expect pre-filter to handle)
    "Joe's Plumbing LLC", "Sunrise Dental", "Apex Roofing Inc.", "Blue Sky Cleaning Services",
    "Riverstone Bakery", "Bright Smile Dentistry", "Evergreen Landscaping LLC", "Summit Auto Repair",
    "Golden Gate Painting", "Lakeside Veterinary Clinic",
    # ALL CAPS (casing judgment)
    "BEYOND WORDS SPEECH THERAPY", "GREENLAND LANDSCAPING", "SUNRISE DENTAL", "ABC MEDICAL GROUP",
    "PRECISION AUTO BODY", "IBM CORPORATION",
    # locations (LLM should strip; pre-filter must DEFER)
    "Joe's Plumbing of Dallas", "Anago of Greater Newark", "Stratus Building Solutions of Tampa",
    "Mr. Handyman of North Austin", "Coastal Cleaning USA",
    # personal names / designators (LLM judgment; pre-filter must DEFER)
    "Lee Mandel & Associates", "Hess Law Firm", "Smith & Sons HVAC", "The Wellington Group",
    "Robert Slayton & Associates", "Johnson Brothers Construction", "Anderson Partners",
    # ambiguous / structural (must DEFER)
    "Bob's Auto dba QuickFix", "ABC Corp (formerly XYZ Inc)", "Store #1234 Cleaning Co",
    "H&R Block", "Tom & Jerry's Pizza",
    # articles / edge
    "The Bagel Shop", "The Maids", "QuickBooks Pro Services",
    # extra simple
    "Mountain View Roofing", "Crystal Clear Windows", "Happy Paws Grooming", "Elite Fitness Studio",
]

def llm_clean(name):
    rsp = client.responses.create(
        model="gpt-5-mini",
        input=[{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": f'RAW: "{name}"\n\nNormalize this company name. Return JSON only.'}],
    )
    txt = rsp.output_text
    try:
        return json.loads(txt).get("canonical", "").strip()
    except Exception:
        return txt.strip()

results = []
for n in NAMES:
    pf = deterministic_prefilter(n)
    pf_out = pf["canonical"] if pf else None
    llm_out = llm_clean(n)
    results.append({
        "raw": n,
        "prefilter_handled": pf is not None,
        "prefilter_output": pf_out,
        "llm_output": llm_out,
        "match": (pf_out == llm_out) if pf_out is not None else None,
    })
    print(f"  {'PF ' if pf else 'LLM'} | {n!r:48} pf={pf_out!r:30} llm={llm_out!r}")

handled = [r for r in results if r["prefilter_handled"]]
matches = [r for r in handled if r["match"]]
print("\n=== SUMMARY ===")
print(f"total names: {len(results)}")
print(f"pre-filter handled (LLM calls avoided): {len(handled)} ({100*len(handled)//len(results)}% coverage)")
print(f"  of those, EXACT match with LLM: {len(matches)}/{len(handled)}")
print(f"  mismatches (need Claude to judge if pre-filter is WRONG or just different):")
for r in handled:
    if not r["match"]:
        print(f"    {r['raw']!r}: pf={r['prefilter_output']!r} vs llm={r['llm_output']!r}")
json.dump(results, open("/tmp/prefilter_eval.json", "w"), indent=2)
print("\nwrote /tmp/prefilter_eval.json")
