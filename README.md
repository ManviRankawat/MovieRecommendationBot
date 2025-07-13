# 🎬 Movie Recommendation Bot

AI-powered movie recommendation agent with tracing and evaluation.

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Environment File
```bash
# Create .env file
touch .env
```

Add your API keys to `.env`:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here  
JUDGMENT_API_KEY=your-judgment-key-here
```

### 3. Get API Keys

**OpenAI** (Required)
- Go to https://platform.openai.com/api-keys
- Create key, copy it

**Tavily** (Required) 
- Go to https://tavily.com
- Sign up, get free API key

**Judgment Labs** (Required)
- Go to https://app.judgmentlabs.ai
- Sign up, go to Settings → API Keys

### 4. Run the Bot
```bash
python movie_agent.py
```

## ✅ Success Check

You should see:
```
🎬 Welcome to the Movie Recommendation Bot! 🎬
Q: What are some of your favorite movie genres?
A: Action, Sci-Fi, and Thriller
...
🍿 Your Personalized Movie Recommendations 🍿
✅ SUCCESS! Agent completed successfully!
```

Then check https://app.judgmentlabs.ai for traces.

## 🔧 Quick Fixes

**Error: "gpt-4 does not exist"**
```bash
# Edit movie_agent.py, change line:
# FROM: ChatOpenAI(model="gpt-4", temperature=0)
# TO:   ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

**Error: "Invalid API key"**
- Check your `.env` file has no extra spaces
- Make sure keys are correct

**Error: "Project limit exceeded"** 
- Change project name in movie_agent.py to something unique

## 📁 Files

- `movie_agent.py` - Main bot
- `requirements.txt` - Dependencies  
- `eval.py` - Evaluation script
- `trace.py` - Tracing example
- `.env` - Your API keys (create this)

## 🎯 That's it!

The bot will:
1. Ask 6 questions about movie preferences
2. Search for movie information
3. Generate personalized recommendations
4. Track everything in Judgment dashboard

Total setup time: ~5 minutes 🚀
