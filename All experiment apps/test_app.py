# test_app.py
import pytest
from unittest.mock import MagicMock, patch
import os
import torch # Needed for torch.tensor in mocks

# Load environment variables for the test environment too
# This ensures os.getenv calls in the functions under test work correctly
from dotenv import load_dotenv
load_dotenv()

# --- IMPORTANT: MOCK THE MODEL LOADING FIRST ---
# This prevents app5.py from trying to load the actual BioBART model and tokenizer
# when it's imported by the test suite, making tests fast and isolated.
# We also need to mock st.cache_resource if app5.py is directly importing st.
# If you move your functions to a separate file (e.g., utils.py) this becomes cleaner.

# Mock Streamlit specific decorators/functions if they are called at import time
# in app5.py (like st.cache_resource)
patch('app5.st.cache_resource', lambda x: x).start()
patch('app5.st.session_state', {}).start() # Mock session state if accessed at import level
patch('app5.st.set_page_config').start() # Mock UI calls if they run at import time
patch('app5.st.markdown').start()
patch('app5.st.expander').start()
patch('app5.st.sidebar').start()
patch('app5.st.tabs').start()


# Now, import the functions from your main application file
# Make sure app5.py exists and contains these functions
from app5 import (
    is_valid_mri_findings,
    generate_impression,
    enhance_with_gpt,
    extract_findings_from_pdf, # We will add a basic mock test for this
)

# --- Fixtures for Mocking ---

# Mock Azure OpenAI client for is_valid_mri_findings and enhance_with_gpt
@pytest.fixture
def mock_azure_openai_client():
    with patch('app5.AzureOpenAI') as MockClient:
        # Configure a mock response for chat.completions.create
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        MockClient.return_value.chat.completions.create.return_value = mock_response
        yield MockClient.return_value

# Mock BioBART tokenizer and model for generate_impression
@pytest.fixture
def mock_biobart_model():
    # Use patch.object to mock specific methods/attributes on the tokenizer/model instances
    # when they are created via from_pretrained.
    with patch('app5.AutoTokenizer.from_pretrained') as MockTokenizerFromPretrained, \
         patch('app5.AutoModelForSeq2SeqLM.from_pretrained') as MockModelFromPretrained:
        
        # Configure the mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.decode.return_value = "mocked raw impression"
        mock_tokenizer_instance.return_value = mock_tokenizer_instance # from_pretrained returns an instance
        MockTokenizerFromPretrained.return_value = mock_tokenizer_instance

        # Configure the mock model instance
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock(
            __getitem__=MagicMock(return_value=[torch.tensor([1, 2, 3])]) # simulate tensor output
        )
        mock_model_instance.to.return_value = mock_model_instance # chainable .to(DEVICE)
        mock_model_instance.return_value = mock_model_instance # from_pretrained returns an instance
        MockModelFromPretrained.return_value = mock_model_instance

        # Ensure that the actual MODEL_DIR constant is used from app5
        # This is because the tokenizer and model are loaded using MODEL_DIR in app5.py
        from app5 import MODEL_DIR
        MockTokenizerFromPretrained.assert_called_with(MODEL_DIR)
        MockModelFromPretrained.assert_called_with(MODEL_DIR, device_map=None, dtype=torch.float32)

        yield (mock_tokenizer_instance, mock_model_instance)


# Mock the unstructured.partition.pdf and tempfile for PDF extraction
@pytest.fixture
def mock_pdf_extraction():
    with patch('app5.partition_pdf') as mock_partition_pdf, \
         patch('app5.tempfile.NamedTemporaryFile') as mock_tempfile, \
         patch('app5.os.remove') as mock_os_remove:
        
        # Simulate PDF content
        mock_partition_pdf.return_value = [
            MagicMock(text="HISTORY: Patient with headache."),
            MagicMock(text="FINDINGS: Small lesion noted. No hydrocephalus."),
            MagicMock(text="IMPRESSION: 1. Small lesion. 2. No hydrocephalus.")
        ]
        
        # Simulate tempfile behavior
        mock_temp_file_obj = MagicMock()
        mock_temp_file_obj.name = "mock_temp_file.pdf"
        mock_temp_file_obj.__enter__.return_value = mock_temp_file_obj
        mock_temp_file_obj.__exit__.return_value = None
        mock_tempfile.return_value = mock_temp_file_obj
        
        yield mock_partition_pdf, mock_tempfile, mock_os_remove

# --- Unit Tests ---

# Test is_valid_mri_findings
def test_is_valid_mri_findings_yes(mock_azure_openai_client):
    mock_azure_openai_client.chat.completions.create.return_value.choices[0].message.content = "YES"
    assert is_valid_mri_findings("This is a valid MRI finding description.") is True

def test_is_valid_mri_findings_no(mock_azure_openai_client):
    mock_azure_openai_client.chat.completions.create.return_value.choices[0].message.content = "NO"
    assert is_valid_mri_findings("This is not an MRI finding, it's a shopping list.") is False

def test_is_valid_mri_findings_api_key_not_set():
    original_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if original_api_key:
        del os.environ["AZURE_OPENAI_API_KEY"]
    
    # After deleting, it should always return True as per your app logic
    assert is_valid_mri_findings("Any text should pass if API key is not set.") is True
    
    if original_api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = original_api_key

# Test generate_impression
def test_generate_impression(mock_biobart_model):
    mock_tokenizer_instance, mock_model_instance = mock_biobart_model
    findings = "Patient has mild degeneration of the lumbar spine at L4-L5."
    min_len = 50
    max_len = 100
    beams = 5
    
    impression = generate_impression(findings, min_len, max_len, beams)
    
    assert impression == "mocked raw impression"
    mock_tokenizer_instance.encode.assert_called_once_with(findings, max_length=1024, truncation=True) # Ensure input is encoded
    mock_model_instance.generate.assert_called_once() # Ensure model.generate was called
    mock_tokenizer_instance.decode.assert_called_once() # Ensure tokenizer.decode was called


# Test enhance_with_gpt
def test_enhance_with_gpt_success(mock_azure_openai_client):
    mock_azure_openai_client.chat.completions.create.return_value.choices[0].message.content = "1. Enhanced summary of findings."
    
    raw_impression = "1. Raw summary."
    original_findings = "Detailed MRI findings about a small lesion and no hydrocephalus."
    
    enhanced_text = enhance_with_gpt(raw_impression, original_findings)
    assert enhanced_text == "1. Enhanced summary of findings."
    
    mock_azure_openai_client.chat.completions.create.assert_called_once()
    call_args, call_kwargs = mock_azure_openai_client.chat.completions.create.call_args
    assert call_kwargs['model'] == os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    assert "FULL FINDINGS" in call_kwargs['messages'][0]['content']
    assert "DRAFT IMPRESSION" in call_kwargs['messages'][0]['content']
    assert raw_impression in call_kwargs['messages'][0]['content']
    assert original_findings in call_kwargs['messages'][0]['content']

# Test extract_findings_from_pdf
def test_extract_findings_from_pdf(mock_pdf_extraction):
    mock_partition_pdf, mock_tempfile, mock_os_remove = mock_pdf_extraction
    
    # Create a mock PDF file object (Streamlit file_uploader returns a BytesIO object)
    mock_pdf_file_obj = MagicMock()
    mock_pdf_file_obj.getvalue.return_value = b"mock pdf content" # Simulate binary content
    
    expected_findings = "Small lesion noted. No hydrocephalus."
    actual_findings = extract_findings_from_pdf(mock_pdf_file_obj)
    
    assert actual_findings == expected_findings
    mock_pdf_file_obj.getvalue.assert_called_once() # Ensure content was read
    mock_tempfile.assert_called_once() # Ensure temp file was created
    mock_partition_pdf.assert_called_once() # Ensure partition_pdf was called
    mock_os_remove.assert_called_once() # Ensure temp file was removed