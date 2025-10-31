# Document Analysis Agent with LangChain and Ollama

A complete document processing and analysis system that combines preprocessing, hybrid classification, entity extraction, and an intelligent LangChain agent for querying documents.

## System Overview

This system provides end-to-end document processing:

1. **Document Preprocessing** - Clean and prepare documents
2. **Hybrid Classification** - Categorize documents using rule-based + LLM approach
3. **Entity Extraction** - Extract named entities from documents
4. **Storage** - Save to SQLite and Milvus databases
5. **Agent Querying** - Natural language queries with LangChain

## Architecture

```
Document Input
    ↓
Preprocessing (Clean, normalize, extract metadata)
    ↓
    ├──────────────────┬──────────────────┐
    ↓                  ↓                  ↓
OCR Text       Document Image      Metadata
    ↓                  ↓
TF-IDF         EfficientNetB0
Vectorizer     (Vision CNN)
    ↓                  ↓
Text Features  Visual Features
(2000-dim)     (256-dim)
    ↓                  ↓
    └────── Concatenate ──────┘
              ↓
       Fusion Layers
       (Dense Network)
              ↓
    Classification Output
    (email/report/scientific)
    ↓
Entity Extraction (NER - type-specific)
    ↓
Storage (SQLite + Milvus with embeddings)
    ↓
LangChain Agent (Query & Analyze)
    ↓
Results
```

## Complete Pipeline Components

### 1. Document Preprocessing
preprocessing component  handle:
- Image :
   noise cancellation  
   border removal  
   enhancing contrast 
   fixing orientation


**Expected Input:**
- Image File

**Output:**
- Clean Iamge ready for ocr

### 2. Hybrid Multimodal Classification

The system uses a **hybrid deep learning model** that combines **vision and text modalities** for robust document classification.

#### Multimodal Architecture

**Two-Branch Neural Network:**

1. **Vision Branch (EfficientNetB0)**
   - Pre-trained on ImageNet for transfer learning
   - Extracts visual features: layout, structure, formatting
   - Captures document appearance and visual patterns
   - GlobalAveragePooling → Dense(256) with dropout
   - Frozen initially, optionally fine-tuned later

2. **Text Branch (TF-IDF + Dense Layers)**
   - TF-IDF vectorization of OCR-extracted text
   - Max 2000 features with n-grams (1-3)
   - Dense(512) → Dense(256) with dropout and batch normalization
   - Captures content, vocabulary, and semantic information

3. **Fusion Layer**
   - Concatenates vision (256) and text (256) features
   - Combined feature vector (512 dimensions)
   - Additional Dense layers for classification
   - Final softmax layer for class probabilities



#### Document Types

**Three Document Categories:**

**Email:**
- Visual: Email header layout, sender/recipient format
- Text: Conversational tone, greetings, signatures
- Combined: Structure + content patterns

**Report:**
- Visual: Formal document structure, sections, tables
- Text: Business vocabulary, technical terms, formal language
- Combined: Professional formatting + content

**Scientific:**
- Visual: Two-column layout, equations, figures, references
- Text: Academic vocabulary, methodology, citations
- Combined: Research paper structure + scholarly content

#### Why Hybrid Multimodal Classification?

**Advantages:**
- **Complementary Information**: Vision captures what text cannot (layout, formatting)
- **Robustness**: Works even with poor OCR quality
- **Higher Accuracy**: Achieves 90%+ accuracy by combining modalities
- **Context Understanding**: Considers both appearance and content
- **Transfer Learning**: Leverages pre-trained ImageNet weights

**Model Architecture:**
```
Document Image                OCR Text
     ↓                           ↓
EfficientNetB0              TF-IDF Vectorizer
     ↓                           ↓
GlobalAvgPool2D            Dense Layers (512→256)
     ↓                           ↓
Dense(256)                  BatchNorm + Dropout
     ↓                           ↓
     └───────── Concatenate ─────┘
                  ↓
          Dense(256) → Dense(128)
                  ↓
            Softmax Output
                  ↓
      Classification (email/report/scientific)
```

**Training Strategy:**
- Phase 1: Train with frozen vision backbone (fast convergence)
- Phase 2: Fine-tune top vision layers if accuracy < 90%
- Data augmentation on images (rotation, shift, zoom, brightness)
- Early stopping and learning rate reduction
- Best model selection based on validation accuracy

**Required Fields by Type:**

| Type | Required Fields |
|------|----------------|
| Email | sender, recipient, subject, date |
| Report | title, date, author, summary |
| Scientific | abstract, methodology, results, references |

### 3. Entity Extraction

Extracts named entities from documents:

**Entity Types:**
- PERSON - People names
- ORG - Organizations
- EMAIL - Email addresses
- DATE - Dates and times
- GPE - Geopolitical entities
- TITLE - Document titles
- CITATION - References (for scientific papers)
- METHOD - Methodologies
- RESULT - Results and findings

**Entity Schema:**
```python
{
    "label": "PERSON",           # Entity type
    "text": "John Doe",          # Entity text
    "start_pos": 45,             # Start position in document
    "end_pos": 53,               # End position
    "confidence": 0.95,          # Extraction confidence
    "method": "spacy"            # Extraction method
}
```

### 4. OCR
 Used PaddleOcr for English Language

### 5. Summarization
Used Ollama Model 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF' for summerization with a different prompt for each document type. 
### 4. Database Storage
Created dual database archeticutre a vector database for semantic search and relational database for querying
#### the vectors is to be generated:
 using ollama as well with model 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
#### SQLite Schema

```sql
-- Documents table
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    summary TEXT,                -- Generated or extracted summary
    date TEXT,                   -- Document date
    classification TEXT,         -- email, report, or scientific
    anomalies TEXT,             -- Any detected anomalies
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Entities table
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    label TEXT NOT NULL,         -- Entity type (PERSON, ORG, etc.)
    text TEXT NOT NULL,          -- Entity text
    start_pos INTEGER,           -- Start position in document
    end_pos INTEGER,             -- End position
    confidence REAL,             -- Confidence score
    method TEXT,                 -- Extraction method used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

#### Milvus Schema

```python
fields = [
    FieldSchema(
        name="id", 
        dtype=DataType.INT64, 
        is_primary=True, 
        auto_id=True,
        description="Primary ID"
    ),
    FieldSchema(
        name="embedding", 
        dtype=DataType.FLOAT_VECTOR, 
        dim=embedding_dim,          # Your embedding dimension
        description="Document embedding vector"
    ),
    FieldSchema(
        name="text",                 # Optional
        dtype=DataType.VARCHAR,
        max_length=65535,
        description="Document text"
    ),
    FieldSchema(
        name="created_at", 
        dtype=DataType.INT64,
        description="UNIX timestamp"
    )
]
```

### 5. LangChain Agent

Queries and analyzes stored documents using natural language.

**Available Tools:**
1. `get_documents_by_time_period` - Find documents by date range
2. `find_documents_missing_information` - Identify incomplete documents
3. `analyze_document_patterns` - Discover patterns across documents
4. `find_duplicate_documents` - Detect duplicate/similar documents

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with your model
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

# Ensure Milvus is running
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```
note: ensure ollama serve is up

## Usage

### Complete Pipeline Overview

The system processes documents through the following stages:

**1. Document Preprocessing**
   - Iamge pre processing

**2. Hybrid Classification**
   - Rule-based pattern matching for initial classification
   - Keyword and structure analysis
   - LLM validation using Ollama for ambiguous cases
   - Confidence scoring and final classification assignment

**3. Entity Extraction**
   - Named Entity Recognition (NER) using spaCy or custom methods
   - Document-type specific entity extraction
   - Entity validation and confidence scoring
   - Position tracking in original text

**4. OCR**
   - PaddleOCr

**5. Summary Generation**
   - Automatic summarization using Ollama
   - Key information extraction
   - Condensed representation of document content

**6. Embedding Generation**
   - Vector representation of document content
   - Use of sentence transformers or similar models
   - Storage preparation for similarity search

**7. Database Storage**
   - SQLite: Store metadata, classification, entities
   - Milvus: Store embeddings for vector similarity search
   - Maintain relationships between documents and entities

**8. LangChain Agent Queries**
   - Natural language question answering
   - Intelligent tool selection based on query type
   - Multi-step reasoning with ReAct framework
   - Structured result presentation



## Database Architecture Diagram
####Dual Database Architecture with LLM Integration
####┌─────────────────────────────────────────────────────────────────────────┐
####│                         DOCUMENT PROCESSING PIPELINE                     │
####│                              (Abstract Layer)                            │
####│  • Preprocessing  • Multimodal Classification  • Entity Extraction      │
####└────────────────────────────┬────────────────────────────────────────────┘
    ####                         │
        ####                     ▼
                    ┌────────────────┐
                    │  Processed     │
                    │  Document      │
                    │  + Metadata    │
                    │  + Entities    │
                    │  + Embedding   │
                    └────┬───────┬───┘
                         │       │
         ┌───────────────┘       └───────────────┐
         │                                       │
         ▼                                       ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    SQLite Database        ┃          ┃   Milvus Vector Database  ┃
┃  (Structured Metadata)    ┃          ┃   (Embeddings & Vectors)  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
         │                                       │
         │                                       │
         ▼                                       ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  documents TABLE        │          │  Collection: "documents"│
├─────────────────────────┤          ├─────────────────────────┤
│ • id (PRIMARY KEY)      │          │ • id (PRIMARY KEY)      │
│ • filename (TEXT)       │          │ • embedding (VECTOR)    │
│ • summary (TEXT)        │          │   - dim: 384/768/1536   │
│ • date (TEXT)           │          │   - metric: L2/COSINE   │
│ • classification (TEXT) │          │ • text (VARCHAR)        │
│   - email               │          │   - max_length: 65535   │
│   - report              │          │   - optional field      │
│   - scientific          │          │ • created_at (INT64)    │
│ • anomalies (TEXT)      │          │   - UNIX timestamp      │
│ • created_at (TIMESTAMP)│          └─────────────────────────┘
│ • updated_at (TIMESTAMP)│                     │
└─────────────────────────┘                     │
         │                                       │
         │ (1-to-many)                          │
         ▼                                       │
┌─────────────────────────┐                     │
│  entities TABLE         │                     │
├─────────────────────────┤                     │
│ • id (AUTO INCREMENT)   │                     │
│ • document_id (FK)      │◄────────────────────┘
│ • label (TEXT)          │          Linked by document id
│   - PERSON, ORG, DATE   │
│   - EMAIL, GPE, etc.    │
│ • text (TEXT)           │
│ • start_pos (INTEGER)   │
│ • end_pos (INTEGER)     │
│ • confidence (REAL)     │
│ • method (TEXT)         │
│   - spacy, regex, etc.  │
│ • created_at (TIMESTAMP)│
└─────────────────────────┘
         │
         │ FOREIGN KEY CONSTRAINT:
         │ document_id → documents.id
         │ ON DELETE CASCADE
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    INDEX STRUCTURE                           │
│  • PRIMARY KEY indexes on id columns                         │
│  • Foreign key index on entities.document_id                 │
│  • Index on documents.classification (for type filtering)    │
│  • Index on documents.date (for temporal queries)           │
│  • Milvus IVF_FLAT/HNSW index on embedding vectors         │
└─────────────────────────────────────────────────────────────┘

LLM Interaction Points
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGCHAIN AGENT (ReAct)                       │
│                      Orchestrates Tool Selection                     │
└────────┬─────────────┬─────────────┬─────────────┬─────────────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────┐
    │ Tool 1 │   │ Tool 2 │   │ Tool 3 │   │  Tool 4  │
    │ Time   │   │Missing │   │Pattern │   │Duplicate │
    │ Period │   │  Info  │   │Analysis│   │Detection │
    └────┬───┘   └────┬───┘   └────┬───┘   └────┬─────┘
         │            │            │            │
         └────────┬───┴────────┬───┴────────────┘
                  │            │
         ┌────────▼────────────▼─────────┐
         │                                │
         ▼                                ▼
┏━━━━━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━━━━━━━┓
┃   READS FROM:    ┃          ┃    READS FROM:     ┃
┃  SQLite Database ┃          ┃ Milvus Vector DB   ┃
┗━━━━━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━━━━━━━┛
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌────────────────────┐
│ QUERY OPERATIONS │          │  VECTOR OPERATIONS │
│                  │          │                    │
│ • SELECT docs    │          │ • similarity_search│
│   WHERE date     │          │   (query_vector,   │
│   BETWEEN        │          │    top_k=10)       │
│                  │          │                    │
│ • SELECT docs    │          │ • get_embedding    │
│   WHERE          │          │   (doc_id)         │
│   classification │          │                    │
│                  │          │ • compute_         │
│ • SELECT *       │          │   similarity_matrix│
│   FROM entities  │          │   (doc_ids[])      │
│   WHERE doc_id   │          │                    │
│                  │          │ METRICS:           │
│ • JOIN docs      │          │ • L2 distance      │
│   WITH entities  │          │ • Cosine similarity│
└──────────────────┘          └────────────────────┘
         │                                │
         └────────┬───────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ COMBINED       │
         │ RESULTS        │
         │ (JSON)         │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Ollama LLM    │
         │  (Llama-3.2)   │
         │                │
         │ • Processes    │
         │   JSON results │
         │ • Generates    │
         │   NL response  │
         │ • Formats      │
         │   answer       │
         └────────────────┘


## Agent Queries

Once documents are processed and stored, use the LangChain agent to query them using natural language.

### Query Capabilities

**1. Time-Based Queries**
   - "Show me all documents from January 2024 to March 2024"
   - "What documents were created last week?"
   - "Find documents from Q1 2024"

**2. Quality Checks**
   - "Which documents are missing critical information?"
   - "Which email documents don't have a sender?"
   - "Find reports without authors"
   - "Show me incomplete scientific papers"

**3. Pattern Analysis**
   - "What patterns can you identify across these documents?"
   - "What are common themes in scientific papers?"
   - "Which entities appear most frequently?"
   - "Analyze temporal patterns in document creation"

**4. Duplicate Detection**
   - "Are there any duplicate documents?"
   - "Find similar documents to document ID 123"
   - "Show me documents with high similarity scores"

**5. Entity-Based Queries**
   - "Find all documents mentioning 'John Doe'"
   - "Which organizations are mentioned most?"
   - "Show me documents with specific entities"

**6. Classification Queries**
   - "How many emails vs reports do we have?"
   - "Show me the distribution of document types"
   - "What percentage of documents are scientific papers?"


## Project Structure

```
.
├── config.py                  # Configuration settings
├── sqlite_db.py               # SQLite services
├── milvus_db.py               # Milvus services
│
├── preprocessing.py           # preprocessing code
├── classification.py          # classification code
├── entity_extraction.py       # entity extraction code
├── project/summary_llm.py     # summrization code
├── document_processor.py      # Complete pipeline integration for document preperation including summrization
├── tools.py                   # LangChain tools
├── agent.py                   # LangChain ReAct agent
├── factory.py                 # Agent factory
├── main.py                    # Entry point and examples
├
└── requirements.txt           # Python dependencies

```
