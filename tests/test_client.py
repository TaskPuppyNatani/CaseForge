"""Tests for the model client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from benchmark_case_generator.client import ClientConfig, ModelClient


class TestClientConfig:
    def test_default_values(self):
        config = ClientConfig()
        
        assert config.base_url == "http://localhost:1234/v1"
        assert config.model == "qwen"
        assert config.api_key is None
        assert config.timeout == 120
    
    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"TEST_API_KEY": "secret123"}):
            config = ClientConfig(api_key_env="TEST_API_KEY")
            assert config.api_key == "secret123"
    
    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"TEST_API_KEY": "env_secret"}):
            config = ClientConfig(api_key="explicit_secret", api_key_env="TEST_API_KEY")
            assert config.api_key == "explicit_secret"


class TestModelClient:
    @pytest.fixture
    def mock_session_class(self):
        with patch("benchmark_case_generator.client.requests.Session") as mock:
            # Create a proper mock session that supports header assignment
            session_instance = Mock()
            session_instance.headers = {}
            mock.return_value = session_instance
            yield mock
    
    @pytest.fixture
    def mock_session(self, mock_session_class):
        yield mock_session_class.return_value
    
    def test_initialization_without_api_key(self):
        config = ClientConfig()
        client = ModelClient(config)
        
        assert client.base_url == "http://localhost:1234/v1"
        assert "Authorization" not in client.session.headers
    
    def test_initialization_with_api_key(self):
        config = ClientConfig(api_key="test_key")
        client = ModelClient(config)
        
        assert client.session.headers.get("Authorization") == "Bearer test_key"
    
    def test_chat_completion_sends_correct_payload(self, mock_session):
        config = ClientConfig()
        client = ModelClient(config)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_session.post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat_completion(messages)
        
        # Verify request was made
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        
        # Check URL
        assert call_args[0][0] == "http://localhost:1234/v1/chat/completions"
        
        # Check payload
        payload = call_args[1]["json"]
        assert payload["model"] == "qwen"
        assert payload["messages"] == messages
        assert payload["temperature"] == config.temperature
        assert payload["max_tokens"] == config.max_tokens
    
    def test_chat_completion_with_custom_params(self, mock_session):
        config = ClientConfig()
        client = ModelClient(config)
        
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_session.post.return_value = mock_response
        
        client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.9,
            max_tokens=100,
        )
        
        payload = mock_session.post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 100

    def test_optional_numeric_values_preserve_explicit_zero(self, mock_session):
        config = ClientConfig(temperature=0.7, max_tokens=4096)
        client = ModelClient(config)
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_session.post.return_value = mock_response

        client.chat_completion(
            messages=[{"role": "user", "content": "Deterministic"}],
            temperature=0.0,
            max_tokens=0,
        )

        payload = mock_session.post.call_args[1]["json"]
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 0
    
    def test_chat_completion_with_response_format(self, mock_session):
        config = ClientConfig()
        client = ModelClient(config)
        
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_session.post.return_value = mock_response
        
        client.chat_completion(
            messages=[{"role": "user", "content": "JSON please"}],
            response_format={"type": "json_object"},
        )
        
        payload = mock_session.post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}
    
    def test_extract_content_success(self):
        config = ClientConfig()
        client = ModelClient(config)
        
        response = {
            "choices": [
                {"message": {"content": "Hello, world!"}}
            ]
        }
        
        content = client.extract_content(response)
        assert content == "Hello, world!"
    
    def test_extract_content_no_choices(self):
        config = ClientConfig()
        client = ModelClient(config)
        
        response = {"choices": []}
        
        with pytest.raises(ValueError, match="No choices"):
            client.extract_content(response)
    
    def test_extract_content_empty_message(self):
        config = ClientConfig()
        client = ModelClient(config)
        
        response = {"choices": [{"message": {}}]}
        
        with pytest.raises(ValueError, match="Empty content"):
            client.extract_content(response)
    
    def test_check_health_success(self, mock_session):
        config = ClientConfig()
        client = ModelClient(config)
        
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        mock_session.post.return_value = mock_response
        
        assert client.check_health() is True
    
    def test_check_health_failure(self, mock_session):
        config = ClientConfig()
        client = ModelClient(config)
        
        mock_session.post.side_effect = Exception("Connection error")
        
        assert client.check_health() is False
    
    def test_base_url_trailing_slash_handling(self):
        config = ClientConfig(base_url="http://example.com/v1/")
        client = ModelClient(config)
        
        assert client.base_url == "http://example.com/v1"
