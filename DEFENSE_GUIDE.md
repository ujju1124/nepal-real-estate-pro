# 🎓 DEFENSE GUIDE: Nepal Real Estate Project
## Simple Explanation for Project Defense

**Student**: Ujjwal Dahal (79010340)  
**Project**: Nepal Land & House Price Prediction System  
**Defense Date**: _________

---

## 📌 QUICK PROJECT SUMMARY (2 minutes)

**What we built:**
A website where people can check house and land prices in Kathmandu Valley using machine learning.

**Why we built it:**
In Nepal, buyers don't know if a house price is fair or not. Our system predicts prices based on 11,706 real listings we collected from Hamrobazar and Lalpurja Nepal websites.

**How it works:**
1. We scraped 13,114 property listings from websites
2. Cleaned the data (removed bad entries)
3. Trained 4 AI models to predict prices
4. Built a website with 4 features: Analytics, Price Prediction, Property Search, and AI Chatbot
5. Deployed it online at HuggingFace Spaces

**Key Achievement:**
- Best model accuracy: 77.7% (General Housing)
- Predicts prices within ±18.8% error on average
- Live website anyone can use: https://ujju33-nepal-real-estate-pro.hf.space

---

## 🔄 THE COMPLETE PROJECT FLOW (Easy Explanation)

Think of this project like cooking a meal in 6 steps:

### STEP 1: Data Collection (Web Scraping)
**What we did:** Downloaded property listings from two websites


**Why this was hard:**
- Hamrobazar loads content dynamically (JavaScript) → Used Selenium (automated browser)
- Lalpurja Nepal is a Nuxt.js app → Used special technique to extract from hidden JSON data

**Results:**
- Hamrobazar: 3,869 land listings
- Lalpurja Nepal: 9,245 listings (817 from Kathmandu)
- Total: 13,114 raw listings

**What the examiner might ask:**
- Q: "Why did you use Selenium instead of just requests?"
- A: "Hamrobazar loads data with JavaScript after page loads. Normal requests library only gets the HTML skeleton. Selenium opens a real browser, waits for JavaScript to load, then extracts the full content."

---

### STEP 2: Data Cleaning (Making Data Usable)
**What we did:** Fixed messy data so computers can understand it

**Problems we found:**
1. **Duplicate listings** → Same property listed multiple times
2. **Inconsistent prices** → "2 Crore 50 Lakh", "25000000", "2.5 Cr" all mean the same
3. **Mixed units** → Land in Ropani, Aana, Bigha, Kattha (converted all to Aana)
4. **Missing values** → Some listings had no bedroom count or price
5. **Typos** → "Kathamndu" instead of "Kathmandu"

**How we fixed it:**
- Removed 1,408 bad/duplicate entries
- Wrote Python code to parse Nepali price strings ("2 Crore 50 Lakh" → 25,000,000)
- Standardized land units to Aana (1 Aana = 342.25 sq ft)
- Filled missing bedroom/bathroom values with median (middle value)


**Final result:** 11,706 clean records (89.3% retention rate)

**What the examiner might ask:**
- Q: "Why not just delete rows with missing values?"
- A: "We would lose too much data. Instead, we filled missing bedroom counts with the median (most common value) because it's safer than guessing. For critical fields like price or location, we did delete those rows."

**File created:** `cleaned_*.csv` files in `data/` folder

---

### STEP 3: Exploratory Data Analysis (Understanding Patterns)
**What we did:** Created charts and statistics to understand the market

**Key findings we discovered:**
1. **Neighborhood is KING** → Location affects price more than anything else
   - Hattisar, Naxal: NPR 1+ Crore per Aana
   - Bhaktapur average: NPR 30-40 Lakh per Aana
   
2. **Airport proximity matters** → Properties near airport cost 40% more

3. **Price distribution is skewed** → Most houses are 20-50 Lakh, but a few luxury ones go up to 10 Crore

4. **Wide roads add value** → 20+ feet road = 15-20% price premium

**Tools we used:**
- Plotly charts (violin plots, histograms, scatter plots)
- Correlation analysis (which features relate to price)
- District comparisons (Kathmandu vs Lalitpur vs Bhaktapur)


**What the examiner might ask:**
- Q: "What was your most surprising finding?"
- A: "Airport distance was the single strongest predictor for land prices. We found a correlation of -0.558, meaning the closer to the airport, the higher the price. This makes sense because Tribhuvan Airport area is a premium location."

**Files created:** Notebooks in `notebooks/02-eda/` folder

---

### STEP 4: Feature Engineering (Creating Smart Inputs)
**What we did:** Created new calculated columns that help models predict better

**Think of it like this:** 
Raw data says "house has 5 bedrooms". Feature engineering creates "bedrooms per bathroom ratio = 2.5" which is more informative.

**Key features we created:**

1. **Log Transformation** (for skewed numbers)
   - Original price: 1 Crore to 10 Crore (huge range)
   - Log price: 7 to 8 (smaller range, easier for model)
   - Why? Models work better with normally distributed data

2. **Target Encoding** (for neighborhoods)
   - Instead of "Baneshwor" as text, we use "average price in Baneshwor = 85 Lakh"
   - Model can now understand neighborhood quality as a number

3. **Luxury Score** (combining amenities)
   - Parking: +1 point
   - Garden: +2 points
   - Modular kitchen: +2 points
   - Solar: +2 points
   - Total luxury score = 0 to 10


4. **Urban Centrality** (proximity to city center)
   - Combines Ring Road distance + Airport distance
   - Properties inside Ring Road get high score

5. **Floor Area Ratio**
   - Built-up area / Land area
   - Shows how efficiently land is used

**What the examiner might ask:**
- Q: "Why did you use log transformation?"
- A: "Real estate prices follow a log-normal distribution—most houses are in the 20-50 Lakh range, but a few luxury properties go up to 10 Crore. Log transformation 'squishes' the big numbers so the model doesn't focus only on expensive houses and ignore cheaper ones."

- Q: "What is target encoding?"
- A: "Instead of telling the model 'this house is in Baneshwor', we tell it 'houses in Baneshwor average 85 Lakh'. The model learns neighborhoods through their average prices, which is much more informative than just neighborhood names."

**Files created:** Notebooks in `notebooks/03-feature-engineering/`

---

### STEP 5: Model Training (Teaching AI to Predict)
**What we did:** Trained 4 machine learning models, one for each property type

**Why 4 models instead of 1?**
Different property types have different data:
- General Housing has 24 features (bedrooms, bathrooms, amenities)
- Lalpurja Land has 29 features (includes GPS distances to hospitals, schools)
- One model can't handle all patterns equally well

**The 4 Models:**


| Model | Algorithm | R² Score | Average Error | Training Data |
|-------|-----------|----------|---------------|---------------|
| 1. General Housing | **XGBoost** | 77.7% | ±18.8% | 2,005 houses |
| 2. Lalpurja Land | **CatBoost** | 74.4% | ±19.1% | 971 plots |
| 3. General Land | **CatBoost** | 61.2% | ±27.4% | 3,250 plots |
| 4. Lalpurja Housing | **CatBoost** | 64.8% | ±23.7% | 1,749 houses |

**Why XGBoost for General Housing?**
- Best for mixed numerical + categorical features
- Handles 24 features efficiently
- Built-in regularization prevents overfitting

**Why CatBoost for others?**
- Naturally handles categorical features (neighborhoods, municipality names)
- Better for smaller datasets
- "Ordered Boosting" technique reduces overfitting

**Training Process:**
1. Split data: 80% training, 20% testing (never show test data to model during training)
2. Hyperparameter tuning: Tried different settings (learning rate, tree depth)
3. Cross-validation: 5-fold CV to ensure model works on unseen data
4. Early stopping: Stop training if model stops improving (prevents overfitting)
5. Save models as .pkl files (pickle format)

**What the examiner might ask:**
- Q: "What is R² score?"
- A: "R² (R-squared) measures how well the model explains price variation. 77.7% means our model explains 77.7% of why prices are different. The remaining 22.3% is due to factors we don't have data for (like view, interior quality, owner urgency)."


- Q: "Why is General Land model accuracy lower (61.2%)?"
- A: "Land valuation is inherently harder than housing because land price depends heavily on intangible factors—development potential, zoning regulations, future road plans—which aren't in our scraped data. Houses have concrete features like bedrooms and bathrooms that are easier to measure."

- Q: "How did you prevent overfitting?"
- A: "Three ways: (1) Train-test split—never tested on training data. (2) Cross-validation—tested on 5 different splits. (3) Regularization—both XGBoost and CatBoost penalize overly complex models."

**Files created:** 
- 4 model files: `xgboost_housing_final.pkl`, `catboost_land_model_final.pkl`, etc.
- Saved in `models/` folder

---

### STEP 6: Application Development (Building the Website)
**What we did:** Created a user-friendly website with 4 sections

**Technology Stack:**
- **Streamlit**: Python web framework (easy to build dashboards)
- **Plotly**: Interactive charts
- **Docker**: Containerization for consistent deployment
- **HuggingFace Spaces**: Free hosting platform

**The 4 Sections Explained:**

---

#### 📊 **Section 1: Market Analytics**
**What it does:** Shows market insights with charts

**Features:**
1. **Overview Page**
   - Total listings: 11,706
   - Average prices by district
   - Price distribution histograms


2. **Housing Market Page**
   - Price vs. bedrooms chart
   - Top 10 most expensive neighborhoods
   - Amenity correlation (does parking increase price?)

3. **Land Market Page**
   - Price per Aana by district
   - Road type impact on price
   - Airport proximity analysis

**Simple explanation for defense:**
"This section helps investors understand market trends. For example, if someone asks 'Which neighborhood is cheapest?', they can see Bhaktapur averages 35 Lakh per Aana vs. Hattisar at 1.2 Crore."

---

#### 🧠 **Section 2: Inference Engine** (Price Prediction)
**What it does:** Predicts price for a property

**How it works (step-by-step):**
1. User selects model (General Housing, General Land, etc.)
2. User fills form:
   - District: Kathmandu
   - Neighborhood: Baneshwor
   - Bedrooms: 3
   - Bathrooms: 2
   - Land: 4 Aana
   - Built-up: 1,200 sq ft
   - Amenities: Parking ✓, Garden ✗, etc.

3. System converts inputs to model format:
   - Neighborhood "Baneshwor" → Encoded value 0.852 (from training data)
   - Calculates luxury_score = 3 (parking + other amenities)
   - Creates feature array: [1, 4, 1200, 2, 0, 16, 3, 2, ..., 0.852]


4. Loads trained model (.pkl file)
5. Makes prediction: NPR 2.85 Crore
6. Shows confidence: High (property is similar to training data)

**Explainability Feature** (Why this price?)
Uses perturbation analysis:
- Changes land from 4 → 4.4 Aana (+10%)
- Checks new prediction: NPR 3.1 Crore (+8.8%)
- Conclusion: "Land size increased price by 8.8%"

Does this for all features, shows top 5 drivers:
1. Neighborhood (Baneshwor): +35% influence
2. Land size: +22% influence
3. Built-up area: +18% influence
4. Bedrooms: +12% influence
5. Road width: +8% influence

**Simple explanation for defense:**
"This is like a real estate agent's brain in code. You describe a property, it predicts the fair price. The explainability feature tells you WHY—like 'This house is expensive mainly because it's in Baneshwor, a premium area.'"

**What the examiner might ask:**
- Q: "What if someone enters a neighborhood not in your training data?"
- A: "The system shows an error message: 'Neighborhood not found in training data. Please select from the dropdown.' We validate inputs before prediction to prevent crashes."

---

#### 🔍 **Section 3: Recommendations** (Property Search)
**What it does:** Finds properties matching user's budget and needs

**How it works:**
1. User sets filters:
   - Budget: 2-4 Crore
   - Bedrooms: 3+
   - District: Kathmandu
   - Must have: Parking, Garden


2. System calculates matching score for each property:
   - **Price score** (30% weight): How close to ideal budget?
     - Property = 3 Cr, Ideal = 3 Cr → Score = 100%
     - Property = 5 Cr, Ideal = 3 Cr → Score = 33%
   
   - **Bedroom score** (20% weight): Matches bedroom count?
     - Has 3 beds, Want 3 beds → Score = 100%
   
   - **Amenity score** (50% weight): Has desired amenities?
     - Has parking ✓ and garden ✓ → Score = 100%
     - Has parking ✓ but no garden ✗ → Score = 50%

3. Ranks all properties by total matching score
4. Shows top 20 results in a table

**Simple explanation for defense:**
"This is like Nepal's version of Zillow or Redfin search filters. Instead of manually scrolling through 11,000 listings, users get the best matches instantly."

---

#### 💬 **Section 4: Property Assistant** (AI Chatbot)
**What it does:** Answers real estate questions using RAG (Retrieval-Augmented Generation)

**What is RAG? (Simple explanation)**
Traditional chatbot: Just makes up answers from memory (can be wrong)  
RAG chatbot: Searches your documents first, then answers based on what it found (more accurate)

**How our RAG system works:**

**Step 1: Build Knowledge Base** (done once at startup)
- We wrote 10 documents covering:
  - Market overview (11,706 listings, 3 districts)
  - Model performance (R² scores, errors)
  - Top neighborhoods by district
  - Investment advice ("Bhaktapur is most affordable")
  - Price ranges per district


- Split each document into 600-character chunks (with 80-char overlap)
- Convert chunks to vectors (embeddings) using HuggingFace model
- Store vectors in FAISS database (like a smart search engine)

**Step 2: User Asks Question**
Example: "What is the median price in Bhaktapur?"

**Step 3: Retrieve Relevant Context**
- Convert question to vector
- FAISS finds top 5 most similar chunks:
  - Chunk 1: "Bhaktapur median land price: 35 Lakh per Aana"
  - Chunk 2: "Bhaktapur is the most affordable district..."
  - Chunk 3: "District comparison: Kathmandu 80L, Lalitpur 65L, Bhaktapur 35L"

**Step 4: Generate Answer**
- Send to GPT-4o-mini (via GitHub Models API):
  ```
  Context: [retrieved chunks]
  Question: What is the median price in Bhaktapur?
  Answer based only on the context above.
  ```
- GPT responds: "According to our market data, Bhaktapur's median land price is 35 Lakh per Aana, making it the most affordable district in the Kathmandu Valley."

**Why RAG instead of just GPT?**
- GPT alone might hallucinate: "Bhaktapur costs 90 Lakh" (wrong!)
- RAG grounds answers in our actual data: Only answers from knowledge base

**Technology Stack:**
- **LangChain**: Orchestrates the RAG pipeline
- **FAISS**: Fast vector search (finds similar chunks in milliseconds)
- **HuggingFace**: Generates embeddings (converts text → numbers)
- **GPT-4o-mini**: Generates natural language answers


**Simple explanation for defense:**
"Imagine asking a real estate expert questions. But instead of a human, it's an AI that searches our 11,706 listings data and answers in natural language. It can't make up facts because it only answers from documents we gave it."

**What the examiner might ask:**
- Q: "What if the chatbot doesn't know the answer?"
- A: "If FAISS doesn't find relevant chunks (similarity score too low), GPT responds: 'I don't have enough information in my knowledge base to answer this question.' It never makes up fake data."

- Q: "Why did you use GitHub Models instead of OpenAI directly?"
- A: "GitHub Models provides free access to GPT-4o-mini for educational projects. OpenAI API would require payment. Both use the same underlying model."

---

## 🚀 DEPLOYMENT (How We Made It Live)

**Challenge:** How do we put this on the internet so anyone can use it?

**Solution:** Docker + HuggingFace Spaces

### What is Docker?
Think of it like a lunchbox:
- You pack your app, Python, all libraries into one container
- Works the same way on any computer (Windows, Mac, Linux, cloud)
- No "but it works on my laptop!" problems

**Our Dockerfile:**
```dockerfile
FROM python:3.12-slim          # Start with Python 3.12
COPY requirements.txt .        # Copy list of libraries
RUN pip install -r requirements.txt  # Install libraries
COPY data/ ./data/            # Copy data files
COPY models/ ./models/        # Copy trained models
COPY app_final.py .           # Copy main app
CMD streamlit run app_final.py  # Start app
```


### What is HuggingFace Spaces?
- Free hosting platform for ML apps
- Runs Docker containers
- Provides public URL: https://ujju33-nepal-real-estate-pro.hf.space

**Deployment Steps:**
1. Create Git repository with all files
2. Track model files using Git LFS (Large File Storage)
   - Models are 2MB total, too big for regular git
3. Add HuggingFace frontmatter to README:
   ```yaml
   ---
   title: Nepal Real Estate Pro
   sdk: docker
   app_port: 7860
   ---
   ```
4. Push to HuggingFace
5. HuggingFace builds Docker image
6. App goes live!

**Why Python 3.12?**
XGBoost 3.3.0 requires Python 3.12+. We initially used 3.11 but upgraded during deployment.

**Simple explanation for defense:**
"We packaged the entire app as a Docker container—like a complete food delivery box with utensils, plates, and food. HuggingFace Spaces hosts it for free, giving us a public URL. Anyone with internet can now use our price predictor."

---

## 🎯 KEY ACHIEVEMENTS

1. **Data Scale**: 11,706 cleaned records from 13,114 raw scrapes
2. **Model Accuracy**: 77.7% R² for housing, ±18.8% error (competitive with international studies)
3. **Deployment**: Live website accessible globally
4. **Innovation**: First ML-based real estate tool for Nepal (no centralized price database exists)
5. **Practical Impact**: Reduces information asymmetry in Nepali real estate market


---

## 🤔 COMMON EXAMINER QUESTIONS & ANSWERS

### Technical Questions

**Q1: Why did you choose XGBoost and CatBoost over other algorithms?**

**A:** "We tried multiple algorithms—Linear Regression, Random Forest, XGBoost, CatBoost. XGBoost gave the best performance for General Housing (77.7% R²) because it handles mixed data types well and has built-in regularization. CatBoost was better for datasets with high-cardinality categorical features like neighborhood names—it has 'Ordered Boosting' that prevents overfitting on small datasets."

---

**Q2: How do you handle missing values?**

**A:** "It depends on the feature:
- For **numerical features** (bedrooms, bathrooms): We used **median imputation** because it's robust to outliers. Mean would be skewed by luxury properties.
- For **critical features** (price, location): We **deleted the row** because you can't predict house price without knowing the price in training data!
- For **categorical features** (road type): We used **mode imputation** (most common value)."

---

**Q3: What is the difference between training data and testing data?**

**A:** "Training data (80%) is what the model learns from. Testing data (20%) is data the model has NEVER seen—we use it to check if the model works on new properties. If we test on training data, the model just 'memorizes' answers, which is cheating. Real accuracy comes from testing on unseen data."

---

**Q4: How do you prevent overfitting?**

**A:** "Three techniques:
1. **Train-test split**: Never test on training data
2. **Cross-validation**: Test on 5 different data splits to ensure consistency

3. **Regularization**: Both XGBoost and CatBoost have L1/L2 penalties that punish overly complex models
4. **Early stopping**: Stop training when validation error stops decreasing"

---

**Q5: Why is your General Land model accuracy lower (61.2%) than housing (77.7%)?**

**A:** "Land valuation is inherently harder because:
1. **Intangible factors**: Development potential, future road plans, zoning—we don't have this data
2. **Speculation**: Land prices include buyer speculation, which is unpredictable
3. **Fewer concrete features**: Houses have bedrooms, bathrooms (measurable). Land just has area and location.
4. **Data quality**: General Land dataset lacks amenity distances (hospital, airport) that Lalpurja Land has, which explains why Lalpurja Land (74.4%) performs better."

---

**Q6: How does your RAG chatbot prevent hallucination?**

**A:** "Traditional LLMs can 'hallucinate' (make up facts). RAG fixes this with grounding:
1. User asks: 'What's the median price in Bhaktapur?'
2. System searches knowledge base FIRST using FAISS vector search
3. Retrieves relevant chunks: 'Bhaktapur median = 35 Lakh per Aana'
4. Gives chunks to GPT with instruction: 'Answer ONLY from this context'
5. GPT can only answer from provided facts, not from its general knowledge

If no relevant chunk is found, GPT says 'I don't have information' instead of guessing."

---

**Q7: What is Git LFS and why did you need it?**

**A:** "Git LFS (Large File Storage) is for tracking big files in git. Regular git is designed for code files (KB), not model files (MB). Our 5 model files total 2MB—too big for regular git.


HuggingFace Spaces also REQUIRES Git LFS for binary files. Without it, our deployment would fail with 'binary file rejected' error."

---

### Project Management Questions

**Q8: How long did this project take?**

**A:** "17 weeks total, split into 6 phases:
- Weeks 1-2: Planning and proposal
- Weeks 3-5: Web scraping (3 weeks because we had to handle dynamic JavaScript)
- Weeks 6-9: Data cleaning and EDA (4 weeks—most time-consuming)
- Weeks 10-12: Model training (3 weeks)
- Weeks 13-15: App development (3 weeks)
- Weeks 16-17: Testing and report writing

Data cleaning took the longest because real-world data is messy—duplicates, typos, mixed units."

---

**Q9: What was the biggest challenge?**

**A:** "Scraping lalpurjanepal.com.np. It's a Nuxt.js SPA (Single Page Application) where content loads via JavaScript after the page loads. 

Initial approach: Use Selenium to render JavaScript (worked but took 30 seconds per listing)

Optimized approach: Discovered Nuxt.js injects data into a hidden `window.__NUXT__` JavaScript object. We extracted this JSON directly, reducing time to 4-6 seconds per listing—5x faster!

This optimization saved us ~8 hours of scraping time for 817 listings."

---

**Q10: How is this project different from existing real estate websites?**

**A:** "Hamrobazar and Lalpurja Nepal only SHOW listings. They don't:
- Predict fair prices
- Provide market analytics
- Offer personalized recommendations
- Have an AI chatbot

Our project adds the 'intelligence layer' on top of listings data, helping buyers answer: 'Is this price fair?' and 'Which properties match my budget?'"


---

### Business/Impact Questions

**Q11: Who is your target audience?**

**A:** "Three groups:
1. **Home buyers**: Check if quoted prices are fair (reduces information asymmetry)
2. **Real estate investors**: Analyze neighborhood trends, compare districts
3. **Researchers**: First publicly available cleaned real estate dataset for Nepal"

---

**Q12: How would you monetize this?**

**A:** "Potential business models:
1. **Freemium**: Basic predictions free, advanced analytics paid (NPR 500/month)
2. **Real estate agent subscription**: Agents pay NPR 2,000/month for unlimited predictions
3. **Data licensing**: Sell cleaned dataset to research institutions
4. **Advertising**: Free for users, revenue from real estate developer ads

Current version is free for educational purposes."

---

**Q13: What are the limitations of your system?**

**A:** "Six key limitations (be honest!):
1. **Data recency**: Scraped in 2026—prices change, need regular updates
2. **Geographic scope**: Only Kathmandu Valley, not Pokhara/Biratnagar
3. **Unexplained variance**: Best model R² = 77.7%, meaning 22.3% of price variation is due to factors we don't capture (view quality, interior condition, owner urgency)
4. **New neighborhoods**: Can't predict for areas not in training data
5. **Scraping dependency**: If source websites change structure, scrapers break
6. **API dependency**: Chatbot needs GitHub Models API (external service)"

---


## 🎤 DEFENSE PRESENTATION STRUCTURE (15 minutes)

### Slide 1: Title (30 seconds)
"Good morning/afternoon. I'm Ujjwal Dahal, presenting 'Nepal Land & House Price Prediction System'—a machine learning platform that predicts real estate prices for the Kathmandu Valley."

---

### Slide 2: Problem Statement (1 minute)
"Nepal's real estate market has a major problem: **information asymmetry**. Buyers don't know if a quoted price is fair. Unlike the US (Zillow), UK (Rightmove), or Singapore (PropertyGuru), Nepal has no transparent price estimation tools. Our project solves this using machine learning."

---

### Slide 3: Data Collection (2 minutes)
"We scraped 13,114 property listings from Hamrobazar and Lalpurja Nepal:
- 3,869 from Hamrobazar (Selenium + BeautifulSoup)
- 9,245 from Lalpurja Nepal (Nuxt.js dual-extraction technique)
- After cleaning: 11,706 records (89.3% retention)
- Covers 3 districts: Kathmandu, Lalitpur, Bhaktapur"

*Show bar chart of raw vs. cleaned data*

---

### Slide 4: Data Pipeline (2 minutes)
"Four-phase pipeline:
1. **Cleaning**: Removed duplicates, standardized units to Aana, parsed Nepali price strings
2. **EDA**: Discovered neighborhood is the #1 price driver, airport proximity matters
3. **Feature Engineering**: Created 15+ derived features—luxury score, urban centrality, floor area ratio
4. **Modeling**: Trained 4 models using XGBoost and CatBoost"

*Show pipeline diagram*

---

### Slide 5: Model Performance (2 minutes)
"Trained 4 models, one per property type:

- General Housing: **77.7% R²**, ±18.8% error (XGBoost)
- Lalpurja Land: **74.4% R²**, ±19.1% error (CatBoost)
- General Land: **61.2% R²**, ±27.4% error (CatBoost)
- Lalpurja Housing: **64.8% R²**, ±23.7% error (CatBoost)

Our best model (77.7%) is competitive with international studies on similar-sized datasets."

*Show actual vs. predicted scatter plot*

---

### Slide 6: Application Demo (4 minutes)
"We built a Streamlit web app with 4 sections:

**1. Market Analytics**: Charts showing price distributions, district comparisons
**2. Inference Engine**: Enter property details → get price prediction + explanation
**3. Recommendations**: Filter by budget, bedrooms, amenities → ranked results
**4. Property Assistant**: RAG-powered chatbot answers questions using FAISS + GPT-4o-mini

*Live demo: Show one prediction with explainability*

Deployed at: https://ujju33-nepal-real-estate-pro.hf.space"

---

### Slide 7: Key Achievements (1 minute)
"Three major contributions:
1. First ML-based real estate price tool for Nepal
2. Created a research-grade dataset (11,706 cleaned records)
3. Achieved 77.7% accuracy on housing prices
4. Deployed publicly accessible tool reducing market information asymmetry"

---

### Slide 8: Limitations & Future Work (1.5 minutes)
"Honest limitations:
- Data from 2026, needs regular updates
- Only Kathmandu Valley
- Land valuation harder (61.2% R²) due to intangible factors

Future enhancements:
- Automated scraping pipeline to keep data fresh
- Expand to Pokhara, Biratnagar
- Integrate GIS data (GPS coordinates, OpenStreetMap)

