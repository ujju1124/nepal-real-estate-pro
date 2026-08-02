# 🎯 How Model Selection Works in Streamlit App

## Simple Explanation for Defense

---

## PART 1: HOW 4 PKL FILES ARE LOADED

### Step 1: At App Startup (Line 280-298)

When the Streamlit app starts, ALL 4 model files are loaded into memory:

```python
def load_models():
    models = {}
    model_files = {
        "gen_house": "models/xgboost_housing_final.pkl",      # General Housing
        "gen_land":  "models/catboost_land_model_final.pkl",  # General Land
        "lph_house": "models/catboost_lalpurja_house_v2_final.pkl",  # Lalpurja Housing
        "lph_land":  "models/catboost_lalpurja_model_final.pkl",     # Lalpurja Land
    }
    for key, fname in model_files.items():
        with open(fname, "rb") as f:
            models[key] = pickle.load(f)
    return models

MODELS = load_models()  # This creates a dictionary with all 4 models
```

### Result:
```python
MODELS = {
    "gen_house": <XGBoost model object>,
    "gen_land": <CatBoost model object>,
    "lph_house": <CatBoost model object>,
    "lph_land": <CatBoost model object>
}
```

**All 4 models are loaded once** when app starts, NOT loaded individually when user selects!

---

## PART 2: HOW USER SELECTS WHICH MODEL TO USE

### In the Inference Engine Section (Line 1280-1325)

User makes 2 choices using radio buttons:

```python
# Choice 1: Property Type
inf_prop_type = st.radio("Property type",
                         ["🏠 House / Building", "🌍 Land / Plot"])

# Choice 2: Has Lalpurja features?
inf_has_lalpurja = st.radio("Advanced features? (Less Reliable)",
                            ["Yes", "No / Not sure"])
```

### Logic to Select Model (Line 1316-1318):

```python
is_house_inf    = "House" in inf_prop_type
is_lalpurja_inf = inf_has_lalpurja == "Yes"

# This determines which model to use:
model_key_inf = ("lph_house" if is_house_inf else "lph_land") if is_lalpurja_inf \
                else ("gen_house" if is_house_inf else "gen_land")
```

### Decision Tree:

```
User selects "House" + "No" → model_key_inf = "gen_house"
User selects "House" + "Yes" → model_key_inf = "lph_house"
User selects "Land" + "No" → model_key_inf = "gen_land"
User selects "Land" + "Yes" → model_key_inf = "lph_land"
```

---


## PART 3: HOW THE SELECTED MODEL IS USED

### When User Clicks "Predict Price" (Example for General Housing)

```python
if model_key_inf == "gen_house":
    # User fills form: district, neighborhood, bedrooms, etc.
    # ... (lines 1329-1380)
    
    # All inputs collected into input_kwargs dictionary:
    input_kwargs = {
        "district": "Kathmandu",
        "neighborhood": "Baneshwor",
        "bedrooms": 3,
        "bathrooms": 2,
        "land_aana": 4,
        "buildup_sqft": 1200,
        # ... more features
    }
    
    # The predict_gen_house() function is called:
    predicted_price = predict_gen_house(**input_kwargs)
```

### Inside predict_gen_house() (Line 361-400):

```python
def predict_gen_house(district, neighborhood, bedrooms, ...):
    # Step 1: Prepare feature array
    row = np.array([[
        MAPS["district"].get(district, 1),
        land_aana,
        buildup_sqft,
        floors,
        # ... all 24 features in correct order
    ]], dtype=np.float32)
    
    # Step 2: Use the SPECIFIC model from MODELS dictionary
    prediction_log = MODELS["gen_house"].predict(row)[0]
    
    # Step 3: Inverse transform (exp - 1)
    predicted_price = float(np.expm1(prediction_log))
    
    return predicted_price
```

### Key Point:
The function `MODELS["gen_house"]` fetches the **XGBoost model** from the dictionary we loaded at startup!

Similarly:
- `MODELS["gen_land"]` → CatBoost land model
- `MODELS["lph_house"]` → CatBoost lalpurja housing model
- `MODELS["lph_land"]` → CatBoost lalpurja land model

---

## 🎓 FOR YOUR DEFENSE

### Examiner Question 1:
**Q: "How do you handle multiple models in your application?"**

**A**: "We load all 4 model files (.pkl) into memory when the app starts using Python's pickle module. They're stored in a dictionary with keys: gen_house, gen_land, lph_house, and lph_land. When a user wants to predict, they select property type (house or land) and whether they have advanced features (Lalpurja data). Based on their selection, we choose the appropriate model from the dictionary and call its predict() method."

---

### Examiner Question 2:
**Q: "Why do you load all 4 models at once? Isn't that memory-intensive?"**

**A**: "It's actually more efficient! Our 4 models total only ~2MB (very small). Loading once at startup means predictions are instant - no waiting for model loading. If we loaded models on-demand, users would wait 1-2 seconds every time they switch models. The memory trade-off is minimal but the UX improvement is significant."

---

### Examiner Question 3:
**Q: "What happens if a user provides data for general housing but accidentally selects Lalpurja?"**

**A**: "The Lalpurja models require additional features like hospital distance, airport distance, etc. If the user selects Lalpurja but doesn't have that data, we show a warning and let them input approximate values or switch back to general models. The UI is designed to guide users - if they select 'No / Not sure' for advanced features, we automatically use the general models which require less information."

---



---

## Summary (Model Selection)

- **4 PKL files loaded** → stored in `MODELS` dictionary at startup
- **User selects** → Property Type (House/Land) + Lalpurja status (Yes/No)
- **Logic determines** → model key based on combination
- **Model used** → retrieved from `MODELS[model_key_inf]` and used for prediction

This makes the system flexible and allows users to get accurate predictions based on their specific property characteristics.

---

# 🤖 PART 2: HOW THE KNOWLEDGE BASE IS BUILT (RAG SYSTEM)

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it like an open-book exam for AI:
- **WITHOUT RAG**: AI only knows what it learned during training (closed-book exam)
- **WITH RAG**: AI can search through documents and use that information to answer (open-book exam)

Your app uses RAG to answer user questions about Nepal's real estate market using **real data from your project**.

---

## Step 1: Create Knowledge Documents (10 Documents)

When the app starts, it creates **10 text documents** containing market insights:

### Document 1: General Housing Statistics
- Total samples: 2,005
- Median price: 3.5 Cr NPR
- Price range: 15 Lakh to 43 Cr
- Top neighborhoods (Old Baneshowr, Thamel, Narayantar)

### Document 2: General Land Statistics
- Total samples: 3,250
- Median price per Ana: 0.49 Cr
- Price correlations (Airport distance: -0.558)
- Road access premium (38% for high access)

### Document 3: Machine Learning Models Info
- Lists all 4 models with their R², accuracy, samples, features
- Example: "General Housing Model (XGBoost), R² score: 0.777, Average error: ±18.8%"

### Document 4: Top 10 Housing Neighborhoods
- Lists most expensive areas with median prices
- Calculated from general housing dataset

### Document 5: Top 10 Land Neighborhoods
- Lists most expensive land areas with price per Ana
- Calculated from general land dataset

### Document 6: Buyer's Guide
- Location importance
- Housing buying tips (bedroom count, Ana size, amenities)
- Land investment tips (small plots in prime locations)
- Price ranges (Budget: <2.45 Cr, Luxury: >4.5 Cr)

### Document 7: District Comparison
- Kathmandu: Highest prices, widest range
- Lalitpur: Moderate prices, cultural heritage
- Bhaktapur: Most affordable, consistent pricing

**NOTE**: Documents 8, 9, 10 are similar insights with more details.

📍 **Code Location**: `app_final.py` lines 580-706

---

## Step 2: Split Documents into Small Chunks

The 10 documents are too long to process efficiently, so they're split into **smaller chunks**:

```
Original Document (1000+ characters)
         ↓
RecursiveCharacterTextSplitter
         ↓
Multiple Chunks (600 characters each)
```

**Settings**:
- `chunk_size=600`: Each chunk is maximum 600 characters
- `chunk_overlap=80`: Chunks overlap by 80 characters to preserve context

**Why overlap?** Prevents sentences from being cut in half between chunks.

**Example**:
```
Chunk 1: "...Airport distance correlation: -0.558..."
         [overlap 80 chars]
Chunk 2: "...correlation: -0.558. Ring Road distance..."
```

📍 **Code Location**: `app_final.py` line 708

---

## Step 3: Convert Chunks to Embeddings (Vector Form)

Each chunk is converted into a **vector** (list of numbers) that represents its meaning.

**Model Used**: `sentence-transformers/all-MiniLM-L6-v2`
- Free, open-source model from HuggingFace
- Runs on CPU (no GPU needed)
- Converts text → 384-dimensional vector

**Example**:
```
Text: "Airport distance is the strongest predictor"
       ↓
Embedding: [0.234, -0.891, 0.456, ..., 0.123] (384 numbers)
```

**Why vectors?** So we can mathematically measure which chunks are **similar** to the user's question.

📍 **Code Location**: `app_final.py` lines 710-714

---

## Step 4: Store Embeddings in FAISS Vector Database

All chunk embeddings are stored in a **FAISS vector database**.

**FAISS** = Facebook AI Similarity Search
- Fast searching through millions of vectors
- Finds most similar chunks to a query in milliseconds

Think of it like:
- **Regular database**: Search by exact keyword match
- **Vector database**: Search by meaning/similarity

📍 **Code Location**: `app_final.py` line 715

---

## Step 5: Build RAG Chain for Question Answering

When a user asks a question, the RAG chain processes it in 4 steps:

### 5.1: User Question → Find Similar Chunks (Retrieval)

```
User asks: "What is the average price of houses in Kathmandu?"
         ↓
Convert question to embedding
         ↓
Search FAISS database for top 5 similar chunks
         ↓
Retrieved chunks: 
  - "Kathmandu median price: 3.6 Cr..."
  - "Top neighborhoods: Old Baneshowr..."
  - "Price range: 15 Lakh to 43 Cr..."
```

**Settings**: `k=5` means retrieve **top 5 most similar chunks**

📍 **Code Location**: `app_final.py` lines 720-723

---

### 5.2: Send Question + Retrieved Chunks to GPT-4o-mini

```
LLM: GPT-4o-mini (via Azure OpenAI)
Temperature: 0.2 (low = more factual, less creative)
```

The AI receives:
1. **System prompt**: Instructions on how to behave
2. **Context**: The 5 retrieved chunks
3. **Question**: User's original question

**System Prompt** (simplified):
```
"You are a Nepal Real Estate Assistant.
Use ONLY the context provided below to answer.
If context doesn't have info, say so honestly.
Format prices in NPR Crore/Lakh."
```

📍 **Code Location**: `app_final.py` lines 725-743

---

### 5.3: GPT-4o-mini Generates Answer

The AI:
1. Reads the 5 retrieved chunks
2. Finds relevant information
3. Generates a natural language answer
4. Formats prices properly (Crore/Lakh)

**Example Response**:
```
"Based on the data, Kathmandu district has a median housing price of 
3.6 Cr NPR. The top neighborhoods include:
• Old Baneshowr: ~5.2 Cr
• Thamel: ~4.8 Cr
• Narayantar: ~4.5 Cr

Prices range from 15 Lakh to 43 Cr depending on location and amenities."
```

📍 **Code Location**: `app_final.py` line 747 (chain execution)

---

### 5.4: Stream Response to User

The answer is **streamed word-by-word** to the user interface:

```
"Based on the data..." [appears]
"Based on the data, Kathmandu..." [appears]
"Based on the data, Kathmandu district..." [appears]
```

This creates a typing effect so users don't wait for the entire response.

---

## Complete RAG Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: BUILD KNOWLEDGE BASE (happens once at startup) │
└─────────────────────────────────────────────────────────┘
                         ↓
    10 Documents → Split into Chunks (600 chars)
                         ↓
    Chunks → Convert to Embeddings (384-dim vectors)
                         ↓
    Embeddings → Store in FAISS vector database

┌─────────────────────────────────────────────────────────┐
│ STEP 2: ANSWER USER QUESTIONS (happens for each query) │
└─────────────────────────────────────────────────────────┘
                         ↓
    User Question: "What's the price in Kathmandu?"
                         ↓
    Convert question to embedding
                         ↓
    Search FAISS for top 5 similar chunks
                         ↓
    Send chunks + question to GPT-4o-mini
                         ↓
    GPT generates answer using ONLY those chunks
                         ↓
    Stream answer to user word-by-word
```

---

## Key Technologies Used

| Technology | Purpose | Location |
|------------|---------|----------|
| **LangChain** | Orchestrates the RAG pipeline | Core framework |
| **sentence-transformers** | Converts text to embeddings | All-MiniLM-L6-v2 model |
| **FAISS** | Vector database for similarity search | Facebook AI library |
| **GPT-4o-mini** | Generates natural language answers | Azure OpenAI API |
| **RecursiveCharacterTextSplitter** | Splits documents into chunks | LangChain utility |

---

## Why This Approach?

### ✅ Advantages:
1. **Accurate**: AI only uses your real project data
2. **Fast**: FAISS retrieves relevant chunks in milliseconds
3. **Transparent**: You know exactly what data the AI is using
4. **Cost-effective**: GPT-4o-mini is cheap (~$0.15 per 1M tokens)
5. **No hallucination**: AI can't make up facts—limited to retrieved chunks

### ⚠️ Limitations:
1. **Limited to knowledge base**: Can't answer questions outside the 10 documents
2. **Chunk quality matters**: If important info is split across chunks, answers may be incomplete
3. **Retrieval dependency**: If wrong chunks are retrieved, answer will be wrong

---

## 🎓 Defense Question Examples (RAG)

### Examiner Question 1:
**Q: "What is RAG and why did you use it?"**

**A**: "RAG stands for Retrieval-Augmented Generation. It's like giving the AI an open-book exam. Instead of only using what it learned during training, it can search through our real estate data documents and provide accurate answers based on our actual project data. This prevents hallucination and ensures all answers are grounded in our research."

---

### Examiner Question 2:
**Q: "How does the knowledge base work?"**

**A**: "We created 10 documents containing all our market insights—housing statistics, land prices, model performance, district comparisons, buyer guides, etc. These are split into 600-character chunks, converted to vectors using sentence-transformers, and stored in a FAISS database. When a user asks a question, we find the 5 most relevant chunks and send them to GPT-4o-mini to generate a natural language answer."

---

### Examiner Question 3:
**Q: "What embedding model did you use?"**

**A**: "We used `all-MiniLM-L6-v2` from HuggingFace. It's a lightweight model that converts text to 384-dimensional vectors and runs efficiently on CPU without needing a GPU. It's perfect for our use case since we have a small knowledge base and need fast, cost-effective embeddings."

---

### Examiner Question 4:
**Q: "Why FAISS instead of a regular database?"**

**A**: "Regular databases search by exact keyword matches. FAISS is a vector database that searches by semantic similarity—it understands meaning. So if a user asks 'expensive areas in Kathmandu,' it finds chunks about 'top neighborhoods' and 'high prices' even if the exact words don't match. This makes the search more intelligent and user-friendly."

---

### Examiner Question 5:
**Q: "What prevents the AI from hallucinating or making up information?"**

**A**: "The system prompt explicitly tells GPT-4o-mini to 'Use ONLY the context provided' and to 'say so honestly if the context doesn't contain enough information.' Since we only pass the 5 retrieved chunks as context, the AI is limited to information from our 10 documents. If a question can't be answered with the available data, it tells the user to explore other sections of the app instead of inventing an answer."

---

## Summary (Knowledge Base & RAG)

- **10 documents** created with market insights from your datasets
- **Split into chunks** (600 chars each, 80 chars overlap)
- **Converted to vectors** using `sentence-transformers/all-MiniLM-L6-v2`
- **Stored in FAISS** vector database for fast similarity search
- **User asks question** → Retrieve top 5 similar chunks → Send to GPT-4o-mini → Generate answer
- **Result**: Accurate, grounded answers using only your real project data

---

## 📊 Quick Comparison: Models vs RAG

| Feature | ML Models (4 PKLs) | RAG Knowledge Base |
|---------|-------------------|-------------------|
| **Purpose** | Price prediction | Answer user questions |
| **Input** | Structured features (bedrooms, Ana, etc.) | Natural language questions |
| **Output** | Numerical price | Text explanation |
| **Data Source** | Trained on 2,000-3,000 samples | 10 documents with insights |
| **Technology** | XGBoost/CatBoost | LangChain + FAISS + GPT-4o-mini |
| **When Used** | Inference Engine section | Property Assistant section |
| **Accuracy Metric** | R² score (0.61 - 0.78) | Quality of retrieved chunks |

Both systems work together to provide:
- **Predictions**: "Your house is worth 3.5 Cr" (Models)
- **Explanations**: "Why is it that price? What should I know?" (RAG)
