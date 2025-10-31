# processors/ner_extractor.py
import re
import spacy
from typing import List, Dict, Tuple
from datetime import datetime
from config import SPACY_MODEL
from logger.logger_config import Logger


log = Logger.get_logger(__name__)

class NERExtractor:
    """
    Named Entity Recognition using spaCy + custom patterns
    Extracts entities specific to each document type
    """
    
    def __init__(self):
        log.info(f"Loading spaCy model: {SPACY_MODEL}...")
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            log.info(f"Model {SPACY_MODEL} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL])
            self.nlp = spacy.load(SPACY_MODEL)
        
        # Add custom patterns
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add custom entity patterns for domain-specific entities"""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Grant number pattern
        grant_pattern = [
            {"TEXT": {"REGEX": r"(?i)grant"}},
            {"TEXT": {"REGEX": r"(?i)(application|no\.?|number)?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"[:#]?"}, "OP": "?"},
            {"TEXT": {"REGEX": r"\w+\d+"}}
        ]
        self.matcher.add("GRANT_NUMBER", [grant_pattern])
        
        # Document ID pattern (like TRAC0901)
        doc_id_pattern = [
            {"TEXT": {"REGEX": r"[A-Z]{3,}\d+"}}
        ]
        self.matcher.add("DOCUMENT_ID", [doc_id_pattern])
    
    def extract_entities(self, text: str, doc_type: str = None) -> List[Dict]:
        """
        Extract all entities from text
        
        Args:
            text: Document text
            doc_type: Document type for specialized extraction
        
        Returns:
            List of entity dictionaries
        """
        doc = self.nlp(text)
        entities = []
        
        # 1. Standard spaCy entities
        UNWANTED_LABELS = {"ORDINAL", "CARDINAL", "PERCENT", "QUANTITY","PRODUCT"}

        for ent in doc.ents:
            if ent.label_ in UNWANTED_LABELS:
                continue  # Skip unwanted entity types

            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.85,
                "method": "spacy"
            })
        
        # 2. Custom pattern matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            entities.append({
                "text": span.text,
                "label": label,
                "start": span.start_char,
                "end": span.end_char,
                "confidence": 0.9,
                "method": "pattern"
            })
        
        # 3. Regex-based entities (emails, money, dates)
        entities.extend(self._extract_regex_entities(text))
        
        # 4. Document-specific entities
        if doc_type:
            entities.extend(self._extract_type_specific(text, doc_type))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x["start"])        
        return entities
    
    def _extract_regex_entities(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Email addresses
        for match in re.finditer(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text):
            entities.append({
                "text": match.group(),
                "label": "EMAIL",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95,
                "method": "regex"
            })
        
        # Money amounts
        for match in re.finditer(r'\$[\d,]+(?:\.\d{2})?', text):
            entities.append({
                "text": match.group(),
                "label": "MONEY",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95,
                "method": "regex"
            })
        
        # Dates (various formats)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 10/26/2001
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # 2001-10-26
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # October 26, 2001
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": "DATE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                    "method": "regex"
                })
        
        # Phone numbers
        for match in re.finditer(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            entities.append({
                "text": match.group(),
                "label": "PHONE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85,
                "method": "regex"
            })
        
        return entities
    
    def _extract_type_specific(self, text: str, doc_type: str) -> List[Dict]:
        """Extract entities specific to document type"""
        entities = []
        
        if doc_type == "SCIENTIFIC":
            # Research subjects - broad categories (not just medical/chemical)
            # Extract nouns that appear after common research patterns
            
            # Common research patterns
            research_patterns = [
                r'research on\s+(\w+(?:\s+\w+){0,2})',
                r'study of\s+(\w+(?:\s+\w+){0,2})',
                r'analysis of\s+(\w+(?:\s+\w+){0,2})',
                r'investigation into\s+(\w+(?:\s+\w+){0,2})',
                r'experiments? (?:on|with|using)\s+(\w+(?:\s+\w+){0,2})',
            ]
            
            for pattern in research_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    subject = match.group(1)
                    # Filter out common words
                    if len(subject) > 3 and subject.lower() not in ['the', 'this', 'that', 'these', 'those']:
                        entities.append({
                            "text": subject,
                            "label": "SUBJECT",
                            "start": match.start(1),
                            "end": match.end(1),
                            "confidence": 0.75,
                            "method": "domain_specific"
                        })
            
            # Research methods/techniques - broad
            method_keywords = [
                # General research methods
                "methodology", "method", "technique", "procedure", "protocol",
                "approach", "analysis", "measurement", "observation", "experiment",
                "test", "trial", "study", "investigation", "survey", "assessment",
                # Statistical/analytical
                "regression", "correlation", "modeling", "simulation", "algorithm",
                # Data collection
                "interview", "questionnaire", "sampling", "recording", "monitoring"
            ]
            
            for keyword in method_keywords:
                for match in re.finditer(rf'\b{keyword}s?\b', text, re.IGNORECASE):
                    entities.append({
                        "text": match.group(),
                        "label": "METHOD",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,
                        "method": "domain_specific"
                    })
            
            # Measurements and quantities (with units)
            measurement_pattern = r'\b\d+(?:\.\d+)?\s*(?:percent|%|degrees?|kg|g|mg|ml|l|cm|mm|m|km|hours?|minutes?|seconds?|years?)\b'
            for match in re.finditer(measurement_pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": "MEASUREMENT",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.85,
                    "method": "domain_specific"
                })
        
        # elif doc_type == Config.DOC_TYPES["EMAIL"]:
        elif doc_type == "EMAIL":
            # Email headers
            header_patterns = ["From", "To", "Cc", "Subject", "Date"]
            for header in header_patterns:
                # Allow indentation + multiline
                # pattern = rf"^\s*{header}:\s*(.+(?:\n(?!\s*\w+:).+)*)"
                pattern = rf"{header}:\s*(.+)"
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    entities.append({
                        "text": value,
                        "label": f"EMAIL_{header.upper()}",
                        "start": match.start(1),
                        "end": match.end(1),
                        "confidence": 0.95,
                        "method": "domain_specific"
                    })
        
        elif doc_type == "REPORT":
            # Report subjects - broad topics (not just medical)
            # Extract from common report structures
            
            subject_patterns = [
                r'(?:project|proposal|grant|report|study)\s+(?:on|about|regarding|concerning)\s+["\']?([^"\'\n]{5,50})["\']?',
                r'(?:title|subject):\s*["\']?([^"\'\n]{5,100})["\']?',
                r'(?:focus|objective|goal|aim)(?:ed)?\s+(?:on|at|to)\s+([^,\.\n]{5,50})',
            ]
            
            for pattern in subject_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    subject = match.group(1).strip()
                    entities.append({
                        "text": subject,
                        "label": "SUBJECT",
                        "start": match.start(1),
                        "end": match.end(1),
                        "confidence": 0.8,
                        "method": "domain_specific"
                    })
            
            # Compliance/regulatory terms - broad
            compliance_keywords = [
                # General compliance
                "compliance", "regulation", "regulatory", "approved", "approval",
                "certified", "certification", "accredited", "accreditation",
                "standard", "guideline", "requirement", "criteria",
                # Legal/governance
                "policy", "procedure", "framework", "governance", "oversight",
                "audit", "review", "assessment", "evaluation"
            ]
            
            for keyword in compliance_keywords:
                for match in re.finditer(rf'\b{keyword}s?\b', text, re.IGNORECASE):
                    entities.append({
                        "text": match.group(),
                        "label": "LAW",  # spaCy's label for regulations/legal terms
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.75,
                        "method": "domain_specific"
                    })
        log.info(f"type specific entities: {entities}")
        
        return entities
    
   
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate/overlapping entities, keeping highest confidence"""
        if not entities:
            return []
        
        # Sort by start position and confidence
        entities.sort(key=lambda x: (x["start"], -x["confidence"]))
        
        unique = []
        seen = set()

        for ent in entities:
            # Create a simple hashable key to detect duplicates
            key = (ent["text"].strip().lower(), ent["label"], ent["method"])
            
            # Skip exact duplicates
            if key in seen:
                continue
            
            # Check if overlaps with existing entities
            overlaps = any(
                ent["start"] < existing["end"] and ent["end"] > existing["start"]
                for existing in unique
            )
            
            if not overlaps:
                unique.append(ent)
                seen.add(key)
        
        return unique

    def get_entity_context(self, text: str, entity: Dict, context_chars: int = 100) -> str:
        """Get surrounding context for an entity"""
        start = max(0, entity["start"] - context_chars)
        end = min(len(text), entity["end"] + context_chars)
        context = text[start:end]
        return context.strip()


# Example usage
if __name__ == "__main__":
    extractor = NERExtractor()
    
    report_text = """
    Kinser, Robin D.
From:
Jill Schultz [jill.schultz@covance.com]
Sent:
Monday, September 30, 2002 10:17 AM
To:
Mary Larson; Mary Westrick; chad.briscoe@mdsps.com; kimberly.prchal@mdsps.com;
bettie.l.nelson@pmusa.com; candace.r.adams@pmusa.com; Hans-
Juergen.Roethig@pmusa.com; Jan.Oey@pmusa.com; mohamadi.sarkar@pmusa.com;
robin.d.kinser@pmusa.com; Shixia.Feng@pmusa.com; Valerie.A.King@pmusa.com
Subject:
Agenda for O9/30/02 Weekly TES Status Meeting
Dbi
TEXT.htm Performance
093002
Markers.xis
Agenda.doc
Attached is the agenda and table of site performance markers
for today's
call.
Jill
Confidentiality Notice: This e-mail transmission
may contain confidential or legally privileged
information that is intended only for the individual
or entity named in the e-mail address. If you are not
the intended recipient, you are hereby notified that
any disclosure, copying, distribution, or reliance
upon the contents of this e-mail is strictly prohibited.
If you have received this e-mail transmission in error,
please reply to the sender, so that we can arrange
for proper delivery, and then please delete the message
from your inbox. Thank you.
2067374252
    """
    

    entities = extractor.extract_entities(report_text, "EMAIL")
    
    print('entities:', entities)
    
    