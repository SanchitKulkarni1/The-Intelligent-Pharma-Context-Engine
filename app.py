"""
Streamlit Frontend for The Intelligent Pharma-Context Engine
A clean, modern UI for pharmaceutical image analysis
"""

import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import pipeline components
from main import run_pipeline, build_document
from schema import PharmaDocument

# Page configuration
st.set_page_config(
    page_title="Pharma-Context Engine",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Entity pills */
    .entity-pill {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
    }
    
    /* Stage indicators */
    .stage-complete {
        color: #38ef7d;
    }
    
    .stage-pending {
        color: #f5576c;
    }
    
    /* JSON display */
    .json-container {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        overflow-x: auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">💊 Intelligent Pharma-Context Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered medicine label extraction, verification, and enrichment</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with info and settings"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/pill.png", width=100)
        st.markdown("## 🔬 About")
        st.markdown("""
        This application processes pharmaceutical images through:
        
        1. **🔍 Detection** - YOLOv8 region detection
        2. **📝 OCR** - PaddleOCR text extraction  
        3. **🧩 Extraction** - Entity parsing
        4. **✅ Verification** - RxNorm matching
        5. **📚 Enrichment** - FDA data lookup
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        use_preprocessing = st.toggle("Enable Image Preprocessing", value=True, 
                                       help="Apply denoising, sharpening for blurry images")
        use_llm = st.toggle("Enable LLM Re-ranking", value=True,
                            help="Use Gemini for ambiguous drug matches")
        
        st.markdown("---")
        st.markdown("## 📊 Supported Formats")
        st.markdown("- JPEG, JPG, PNG")
        st.markdown("- Medicine bottles")
        st.markdown("- Blister strips")
        st.markdown("- Prescription labels")
        
        return use_preprocessing, use_llm


def display_upload_section():
    """Display the image upload section"""
    st.markdown("### 📤 Upload Medicine Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a medicine bottle, blister strip, or prescription label"
        )
    
    with col2:
        st.markdown("**💡 Tips for best results:**")
        st.markdown("- Good lighting")
        st.markdown("- Clear, focused image")
        st.markdown("- Visible text and barcode")
    
    return uploaded_file


def display_image_preview(uploaded_file):
    """Display the uploaded image"""
    st.markdown("### 🖼️ Image Preview")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded Image")


def display_processing_stages():
    """Display processing stage indicators"""
    stages = ["Detection", "Barcode", "OCR", "Entities", "Verification", "Enrichment"]
    cols = st.columns(6)
    
    for i, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 1.5rem;">{"🔄" if i == 0 else "⏳"}</div>
                <div style="font-size: 0.8rem; color: #666;">{stage}</div>
            </div>
            """, unsafe_allow_html=True)


def display_results(doc: PharmaDocument):
    """Display the analysis results"""
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ocr_tokens = len(doc.raw_ocr.tokens) if doc.raw_ocr else 0
        st.metric("OCR Tokens", ocr_tokens, help="Number of text regions detected")
    
    with col2:
        has_barcode = "✅ Yes" if doc.barcode else "❌ No"
        st.metric("Barcode", has_barcode)
    
    with col3:
        match_score = doc.verification.match_score if doc.verification and doc.verification.match_score else 0
        st.metric("Match Score", f"{match_score:.0%}")
    
    with col4:
        has_enrichment = "✅ Yes" if doc.enrichment and (doc.enrichment.storage_requirements or doc.enrichment.safety_warnings) else "❌ No"
        st.metric("Enriched", has_enrichment)
    
    # Main results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 OCR Text", "🧩 Entities", "✅ Verification", "📚 Enrichment", "📄 Raw JSON"])
    
    with tab1:
        display_ocr_results(doc)
    
    with tab2:
        display_entity_results(doc)
    
    with tab3:
        display_verification_results(doc)
    
    with tab4:
        display_enrichment_results(doc)
    
    with tab5:
        display_json_output(doc)


def display_ocr_results(doc: PharmaDocument):
    """Display OCR extraction results"""
    if not doc.raw_ocr:
        st.warning("No OCR results available")
        return
    
    st.markdown("#### Extracted Text")
    st.info(doc.raw_ocr.full_text if doc.raw_ocr.full_text else "No text detected")
    
    if doc.raw_ocr.tokens:
        st.markdown("#### Token Details")
        
        token_data = []
        for t in doc.raw_ocr.tokens:
            token_data.append({
                "Text": t.text,
                "Confidence": f"{t.confidence:.1%}",
                "Position": f"({t.bbox[0]}, {t.bbox[1]})"
            })
        
        st.dataframe(token_data, use_container_width=True)


def display_entity_results(doc: PharmaDocument):
    """Display extracted entities"""
    if not doc.extracted_entities:
        st.warning("No entities extracted")
        return
    
    entities = doc.extracted_entities
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💊 Drug Name")
        if entities.drug_name:
            st.success(f"**{entities.drug_name.value}** (confidence: {entities.drug_name.confidence:.0%})")
        else:
            st.warning("Not detected")
        
        st.markdown("#### 📦 Dosage")
        if entities.dosage:
            st.success(f"**{entities.dosage.value}** (confidence: {entities.dosage.confidence:.0%})")
        else:
            st.warning("Not detected")
    
    with col2:
        st.markdown("#### 🏭 Manufacturer")
        if entities.manufacturer:
            st.success(f"**{entities.manufacturer.value}** (confidence: {entities.manufacturer.confidence:.0%})")
        else:
            st.warning("Not detected")
        
        st.markdown("#### 🧪 Composition")
        if entities.composition:
            ingredients = entities.composition.value
            if isinstance(ingredients, list):
                st.success(f"**{', '.join(ingredients)}** (confidence: {entities.composition.confidence:.0%})")
            else:
                st.success(f"**{ingredients}** (confidence: {entities.composition.confidence:.0%})")
        else:
            st.warning("Not detected")


def display_verification_results(doc: PharmaDocument):
    """Display RxNorm verification results"""
    if not doc.verification or not doc.verification.matched_term:
        st.warning("No RxNorm match found")
        return
    
    v = doc.verification
    
    st.markdown("#### ✅ Verified Drug Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    border-radius: 15px; padding: 20px; color: white;">
            <h3 style="margin: 0; color: white;">{v.matched_term}</h3>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">RxNorm CUI: {v.rxnorm_cui}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Match Score", f"{v.match_score:.0%}" if v.match_score else "N/A")
    
    if v.justification:
        st.markdown("#### 🤖 AI Reasoning")
        st.info(v.justification)
    
    # Barcode info
    if doc.barcode:
        st.markdown("#### 📊 Barcode Information")
        st.markdown(f"- **Value:** `{doc.barcode.value}`")
        st.markdown(f"- **Type:** {doc.barcode.symbology}")


def display_enrichment_results(doc: PharmaDocument):
    """Display FDA enrichment data"""
    if not doc.enrichment:
        st.warning("No enrichment data available")
        return
    
    e = doc.enrichment
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌡️ Storage Requirements")
        if e.storage_requirements:
            st.info(e.storage_requirements)
        else:
            st.markdown("_Not specified_")
        
        st.markdown("#### ⚠️ Safety Warnings")
        if e.safety_warnings:
            for warning in e.safety_warnings[:3]:  # Limit to 3
                st.warning(warning[:200] + "..." if len(warning) > 200 else warning)
        else:
            st.markdown("_No warnings available_")
    
    with col2:
        st.markdown("#### 💉 Common Side Effects")
        if e.common_side_effects:
            for effect in e.common_side_effects[:5]:  # Limit to 5
                st.markdown(f"• {effect[:100]}..." if len(effect) > 100 else f"• {effect}")
        else:
            st.markdown("_None listed_")


def display_json_output(doc: PharmaDocument):
    """Display raw JSON output"""
    st.markdown("#### Complete JSON Output")
    st.json(json.loads(doc.model_dump_json()))
    
    # Download button
    st.download_button(
        label="📥 Download JSON",
        data=doc.model_dump_json(indent=2),
        file_name=f"pharma_analysis_{doc.document_id[:8]}.json",
        mime="application/json"
    )


def process_image(uploaded_file, use_preprocessing: bool):
    """Process the uploaded image through the pipeline"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Run pipeline with progress
        with st.spinner("🔍 Detecting regions..."):
            doc = run_pipeline(tmp_path, output_json=False)
        
        return doc
    
    finally:
        # Cleanup
        os.unlink(tmp_path)


def main():
    """Main application"""
    display_header()
    use_preprocessing, use_llm = display_sidebar()
    
    uploaded_file = display_upload_section()
    
    # Handle uploaded file
    if uploaded_file is not None:
        display_image_preview(uploaded_file)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button("🚀 Analyze Image", type="primary", use_container_width=True)
        
        if process_button:
            with st.status("Processing image...", expanded=True) as status:
                st.write("🔍 Detecting regions with YOLO...")
                st.write("📝 Extracting text with OCR...")
                st.write("🧩 Parsing entities...")
                st.write("✅ Verifying against RxNorm...")
                st.write("📚 Enriching with FDA data...")
                
                doc = process_image(uploaded_file, use_preprocessing)
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            # Store in session state
            st.session_state['result'] = doc
            st.session_state['analyzed_image'] = "uploaded"
    
    else:
        # Demo section - Sample Images
        st.markdown("---")
        st.markdown("### 🎯 Try with Sample Images")
        
        sample_images = {
            "barcode1.jpeg": "Diltiazem HCI 30mg Tablet",
            "testimage.jpeg": "Hydrocodone/APAP Prescription",
            "test2.jpeg": "Morphine Sulfate Ampule"
        }
        
        cols = st.columns(3)
        for i, (img_name, description) in enumerate(sample_images.items()):
            with cols[i]:
                img_path = f"images/{img_name}"
                if os.path.exists(img_path):
                    st.image(img_path, caption=description, use_container_width=True)
                    if st.button(f"Analyze", key=img_name):
                        st.session_state['processing'] = img_path
                        st.session_state['processing_name'] = description
        
        # Process sample image if button was clicked
        if 'processing' in st.session_state:
            img_path = st.session_state['processing']
            description = st.session_state.get('processing_name', 'Sample Image')
            
            st.markdown("---")
            st.markdown(f"### 🖼️ Analyzing: {description}")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img_path, use_container_width=True)
            
            with st.status("Processing image...", expanded=True) as status:
                st.write("🔍 Detecting regions with YOLO...")
                st.write("📝 Extracting text with OCR...")
                st.write("🧩 Parsing entities...")
                st.write("✅ Verifying against RxNorm...")
                st.write("📚 Enriching with FDA data...")
                
                doc = run_pipeline(img_path, output_json=False)
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            st.session_state['result'] = doc
            st.session_state['analyzed_image'] = img_path
            
            # Clear processing state
            del st.session_state['processing']
            if 'processing_name' in st.session_state:
                del st.session_state['processing_name']
    
    # Always display results if available
    if 'result' in st.session_state and st.session_state['result'] is not None:
        display_results(st.session_state['result'])


if __name__ == "__main__":
    main()

