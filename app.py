from pathlib import Path
from collections import Counter
import io, re
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
app = FastAPI(title="Apex Pulse API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

DEMO = [
("The strategy call was absolutely perfect today","Race strategy", "positive","Monaco_GP"),
("That penalty makes no sense at all","Penalties & stewarding","negative","Saudi_Arabia_GP"),
("What a recovery drive from the back of the grid","Driver reactions","positive","British_GP"),
("The new regulations have made racing closer","Regulations & technical","positive","Bahrain_GP"),
("Ferrari threw away another podium with the pit stop","Team performance","negative","Italian_GP"),
("Incredible battle through the final sector","On-track battles","positive","Japanese_GP"),
("The stewards need to be more consistent","Penalties & stewarding","negative","Saudi_Arabia_GP"),
("The tire degradation changed everything","Race strategy","neutral","Spanish_GP"),
("Best race of the season so far","Driver reactions","positive","Brazilian_GP"),
("The midfield battle was more exciting than the podium","On-track battles","positive","Australian_GP"),
("The car has serious straight line speed","Regulations & technical","neutral","Belgian_GP"),
("Another questionable team radio message","Team performance","negative","Dutch_GP"),
("Clean overtake, perfectly timed","On-track battles","positive","Canadian_GP"),
("The safety car completely changed the race","Race strategy","neutral","Singapore_GP"),
("That was a brilliant defensive drive","Driver reactions","positive","Miami_GP"),
("Track limits are ruining the spectacle","Penalties & stewarding","negative","Austrian_GP"),
("The upgrades clearly improved the car","Regulations & technical","positive","Emilia_Romagna_GP"),
("Why did they leave him out on old tires","Race strategy","negative","Qatar_GP"),
("The team orders were understandable","Team performance","neutral","Hungarian_GP"),
("What a finish, wheel to wheel racing","On-track battles","positive","Abu_Dhabi_GP"),
]

def load_df():
    for name in ["f1_comments_with_llm_labels.csv","f1_comments_with_topics.csv","f1_comments_cleaned.csv","f1_youtube_comments.csv","f1_reddit_comments.csv"]:
        p=DATA/name
        if p.exists():
            try:
                df=pd.read_csv(p).fillna("")
                if len(df): return normalize_df(df)
            except Exception: pass
    return pd.DataFrame(DEMO,columns=["comment","topic","sentiment","race"])

def normalize_df(df):
    cols={str(c).lower().strip():c for c in df.columns}
    textcol=next((cols[k] for k in ["comment","text","cleaned_text","body","content"] if k in cols),None)
    if textcol is None: textcol=df.columns[0]
    out=pd.DataFrame({"comment":df[textcol].astype(str)})
    topiccol=next((cols[k] for k in ["topic_label","topic_name","llm_topic","topic","dominant_topic","bertopic_topic_raw"] if k in cols),None)
    out["topic"]=df[topiccol].astype(str) if topiccol else "Unclassified"
    sentcol=next((cols[k] for k in ["sentiment","llm_sentiment","sentiment_label"] if k in cols),None)
    out["sentiment"]=df[sentcol].astype(str).str.lower() if sentcol else "neutral"
    racecol=next((cols[k] for k in ["race","race_name","grand_prix","gp"] if k in cols),None)
    out["race"]=df[racecol].astype(str) if racecol else "Unknown GP"
    return out

def payload(df):
    df=df.fillna("")
    topics=Counter(df.topic.astype(str))
    sentiments=Counter(df.sentiment.astype(str).str.lower().map(lambda s: "positive" if "pos" in s else ("negative" if "neg" in s else ("neutral" if "neu" in s else s))))
    races=sorted(set(df.race.astype(str)))
    return {"total":len(df),"race_count":len(races),"topic_count":len(topics),
      "sentiments":dict(sentiments),"topics":[{"name":str(k),"count":int(v)} for k,v in topics.most_common()],
      "races":races,"comments": [{"comment":str(r.comment),"topic":str(r.topic),"sentiment":str(r.sentiment),"race":str(r.race)} for r in df.head(500).itertuples(index=False)],
      "source":"project dataset" if len(df) and not df.comment.isin([x[0] for x in DEMO]).all() else "illustrative demo data"}

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT/"static/index.html").read_text()

@app.get("/api/health")
def health(): return {"status":"ok","service":"Apex Pulse API"}

@app.get("/api/dashboard")
def dashboard(): return payload(load_df())

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Please upload a CSV file.")
    raw=await file.read()
    if len(raw)>20*1024*1024: raise HTTPException(413,"CSV must be smaller than 20 MB.")
    try: df=pd.read_csv(io.BytesIO(raw))
    except Exception as e: raise HTTPException(400,f"Could not read CSV: {e}")
    if df.empty: raise HTTPException(400,"The CSV contains no rows.")
    normalized=normalize_df(df)
    normalized.to_csv(DATA/"uploaded_comments.csv",index=False)
    return {"message":f"Imported {len(normalized):,} comments","dashboard":payload(normalized)}

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
def analyze(body: AnalyzeRequest):
    text=body.text.strip()
    if not text: raise HTTPException(400,"Enter a comment first.")
    # Lightweight transparent baseline; not a trained sentiment model.
    words=set(re.findall(r"[a-z']+",text.lower()))
    pos={"great","good","brilliant","perfect","love","amazing","incredible","best","excellent","exciting","clean","improved","beautiful","win"}
    neg={"bad","terrible","awful","penalty","wrong","hate","poor","worst","slow","ruined","questionable","mistake","disappointing","crash"}
    score=len(words&pos)-len(words&neg)
    sentiment="positive" if score>0 else "negative" if score<0 else "neutral"
    topic_terms={
      "Race strategy":{"strategy","pit","tire","tyre","safety","stop","degradation"},
      "Penalties & stewarding":{"penalty","steward","track","limits","rules"},
      "Driver reactions":{"driver","drive","overtake","defensive","talent"},
      "Team performance":{"team","ferrari","mercedes","red","bull","radio","orders"},
      "Regulations & technical":{"regulation","car","upgrade","aero","technical","speed"},
      "On-track battles":{"battle","racing","wheel","wheel","fight","sector"}
    }
    low=text.lower()
    topic=max(topic_terms,key=lambda t:sum(term in low for term in topic_terms[t]))
    return {"sentiment":sentiment,"topic":topic,"note":"Quick keyword baseline for demonstration; use your trained LDA/BERTopic and Gemini pipeline for research-grade predictions."}
