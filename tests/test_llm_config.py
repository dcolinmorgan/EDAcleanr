"""Tests for LLM configuration module."""

from unittest.mock import patch, MagicMock

import pytest

from src.llm_config import get_llm, DEFAULT_MODELS, SUPPORTED_PROVIDERS


class TestGetLlm:
    """Unit tests for get_llm function."""

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider: 'badprovider'"):
            get_llm(provider="badprovider")

    def test_unsupported_provider_case_insensitive(self):
        with pytest.raises(ValueError):
            get_llm(provider="BADPROVIDER")

    def test_supported_providers_set(self):
        assert SUPPORTED_PROVIDERS == {"openai", "anthropic", "groq", "bedrock"}

    def test_default_models(self):
        assert DEFAULT_MODELS["openai"] == "gpt-4o-mini"
        assert DEFAULT_MODELS["anthropic"] == "claude-3-5-sonnet-20241022"
        assert DEFAULT_MODELS["groq"] == "llama-3.1-70b-versatile"
        assert DEFAULT_MODELS["bedrock"] == "eu.amazon.nova-pro-v1:0"

    @patch("src.llm_config.ChatOpenAI", create=True)
    def test_openai_provider_default_model(self, mock_cls):
        """OpenAI provider uses ChatOpenAI with default model."""
        # Patch at the point of import inside get_llm
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("langchain_openai.ChatOpenAI", mock_cls):
            result = get_llm(provider="openai")

        mock_cls.assert_called_once_with(model="gpt-4o-mini")
        assert result is mock_instance

    @patch("src.llm_config.ChatOpenAI", create=True)
    def test_openai_provider_custom_model(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("langchain_openai.ChatOpenAI", mock_cls):
            result = get_llm(provider="openai", model="gpt-4o")

        mock_cls.assert_called_once_with(model="gpt-4o")
        assert result is mock_instance

    @patch("src.llm_config.ChatAnthropic", create=True)
    def test_anthropic_provider_default_model(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            result = get_llm(provider="anthropic")

        mock_cls.assert_called_once_with(model="claude-3-5-sonnet-20241022")
        assert result is mock_instance

    @patch("src.llm_config.ChatGroq", create=True)
    def test_groq_provider_default_model(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("langchain_groq.ChatGroq", mock_cls):
            result = get_llm(provider="groq")

        mock_cls.assert_called_once_with(model="llama-3.1-70b-versatile")
        assert result is mock_instance

    def test_provider_case_insensitive_valid(self):
        """Provider string is lowercased, so 'OpenAI' should work."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm(provider="OpenAI")
            mock_cls.assert_called_once_with(model="gpt-4o-mini")

    @patch("langchain_openai.ChatOpenAI")
    def test_kwargs_passed_through(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="openai", temperature=0.5, max_tokens=100)
        mock_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.5, max_tokens=100)

    @patch("langchain_aws.ChatBedrock")
    def test_bedrock_provider_default_model(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = get_llm(provider="bedrock")
        mock_cls.assert_called_once_with(model_id="eu.amazon.nova-pro-v1:0")
        assert result is mock_instance

    @patch("langchain_aws.ChatBedrock")
    def test_bedrock_provider_custom_model(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = get_llm(provider="bedrock", model="amazon.titan-text-express-v1")
        mock_cls.assert_called_once_with(model_id="amazon.titan-text-express-v1")
        assert result is mock_instance
