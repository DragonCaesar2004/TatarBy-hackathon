import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import asyncio
import httpx
from fastapi import HTTPException
from main import app, translate_tatsoft, gigachat_complete

client = TestClient(app)

def test_health_endpoint():
    """Test the health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_translate_tatsoft_success():
    """Test successful translation with TatSoft"""
    # Mock the httpx.AsyncClient.get method
    with patch("main.client.get") as mock_get:
        # Create a mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Make json() a regular method that returns the expected data
        mock_response.json = Mock(return_value={"result": "translated text"})
        mock_get.return_value = mock_response
        
        # Call the function
        result = await translate_tatsoft("test text", "tat2rus")
        
        # Assertions
        assert result == "translated text"
        mock_get.assert_called_once()

@pytest.mark.asyncio
async def test_translate_tatsoft_4xx_error():
    """Test TatSoft translation with 4xx error"""
    with patch("main.client.get") as mock_get:
        # Create a mock response with 400 error
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_get.return_value = mock_response
        
        # Check that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await translate_tatsoft("test text", "tat2rus")
        
        # Assertions
        assert exc_info.value.status_code == 400

@pytest.mark.asyncio
async def test_translate_tatsoft_unexpected_format():
    """Test TatSoft translation with unexpected response format"""
    with patch("main.client.get") as mock_get:
        # Create a mock response with missing "result" key
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Make json() a regular method that returns the expected data
        mock_response.json = Mock(return_value={"unexpected": "format"})
        mock_get.return_value = mock_response
        
        # Check that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await translate_tatsoft("test text", "tat2rus")
        
        # Assertions
        assert exc_info.value.status_code == 502

@pytest.mark.asyncio
async def test_gigachat_complete_success():
    """Test successful completion with GigaChat"""
    with patch("main.client.post") as mock_post:
        # Create a mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Make json() a regular method that returns the expected data
        mock_response.json = Mock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "GigaChat response"
                    }
                }
            ]
        })
        mock_post.return_value = mock_response
        
        # Call the function
        result = await gigachat_complete("user prompt", "system prompt")
        
        # Assertions
        assert result == "GigaChat response"
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_gigachat_complete_4xx_error():
    """Test GigaChat completion with 4xx error"""
    with patch("main.client.post") as mock_post:
        # Create a mock response with 400 error
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        # Check that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await gigachat_complete("user prompt", "system prompt")
        
        # Assertions
        assert exc_info.value.status_code == 400

@pytest.mark.asyncio
async def test_gigachat_complete_unexpected_format():
    """Test GigaChat completion with unexpected response format"""
    with patch("main.client.post") as mock_post:
        # Create a mock response with missing structure
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Make json() a regular method that returns the expected data
        mock_response.json = Mock(return_value={"unexpected": "format"})
        mock_post.return_value = mock_response
        
        # Check that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await gigachat_complete("user prompt", "system prompt")
        
        # Assertions
        assert exc_info.value.status_code == 502

@patch("main.translate_tatsoft")
@patch("main.gigachat_complete")
def test_chat_endpoint_success(mock_gigachat, mock_translate):
    """Test successful chat endpoint"""
    # Configure mocks
    mock_translate.side_effect = ["translated to ru", "translated back to tat"]
    mock_gigachat.return_value = "GigaChat response"
    
    # Send request to chat endpoint
    response = client.post("/chat", json={
        "message_tat": "Сәлам!",
        "system_prompt_ru": "You are a helpful assistant."
    })
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["input_tat"] == "Сәлам!"
    assert data["translated_to_ru"] == "translated to ru"
    assert data["model_answer_ru"] == "GigaChat response"
    assert data["translated_back_to_tat"] == "translated back to tat"
    
    # Check that mocks were called correctly
    assert mock_translate.call_count == 2
    mock_gigachat.assert_called_once_with("translated to ru", "You are a helpful assistant.")

@patch("main.translate_tatsoft")
def test_chat_endpoint_translation_error(mock_translate):
    """Test chat endpoint when translation fails"""
    # Configure mock to raise an HTTPException
    mock_translate.side_effect = HTTPException(status_code=502, detail="Translation error")
    
    # Send request to chat endpoint
    response = client.post("/chat", json={
        "message_tat": "Сәлам!",
        "system_prompt_ru": "You are a helpful assistant."
    })
    
    # Assertions
    assert response.status_code == 502
    
    # Check that mock was called
    mock_translate.assert_called_once_with("Сәлам!", "tat2rus")

if __name__ == "__main__":
    pytest.main([__file__])
