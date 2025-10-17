"""Unit tests for botds.llm module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from botds.context import DecisionLog
from botds.llm import LLMRouter


class TestLLMRouterInitialization:
    """Test LLMRouter initialization."""

    def test_init_without_api_key(self, tmp_path, monkeypatch):
        """Test that initialization fails without OpenAI API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
        config = {"openai_model": "gpt-4o-mini"}
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            LLMRouter(config, decision_log)

    def test_init_with_api_key(self, tmp_path, monkeypatch):
        """Test successful initialization with API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI") as mock_openai:
            decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
            config = {"openai_model": "gpt-4o-mini", "ollama_model": "llama3.2"}
            
            router = LLMRouter(config, decision_log)
            
            assert router.openai_model == "gpt-4o-mini"
            assert router.ollama_model == "llama3.2"
            mock_openai.assert_called_once()

    def test_init_checks_ollama_availability(self, tmp_path, monkeypatch):
        """Test that initialization checks Ollama availability."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI"):
            with patch("botds.llm.requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                
                decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
                config = {"openai_model": "gpt-4o-mini"}
                
                router = LLMRouter(config, decision_log)
                
                assert router.ollama_available is True

    def test_init_handles_ollama_unavailable(self, tmp_path, monkeypatch):
        """Test that initialization handles Ollama being unavailable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI"):
            with patch("botds.llm.requests.get") as mock_get:
                mock_get.side_effect = Exception("Connection refused")
                
                decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
                config = {"openai_model": "gpt-4o-mini"}
                
                router = LLMRouter(config, decision_log)
                
                assert router.ollama_available is False


class TestOpenAIDecide:
    """Test OpenAI decision-making functionality."""

    @pytest.fixture
    def router(self, tmp_path, monkeypatch):
        """Create a router with mocked OpenAI client."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
            config = {"openai_model": "gpt-4o-mini"}
            
            router = LLMRouter(config, decision_log)
            router.openai_client = mock_client
            
            yield router

    def test_openai_decide_simple_prompt(self, router):
        """Test simple decision without tools."""
        # Mock response
        mock_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Use RandomForest classifier",
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        router.openai_client.chat.completions.create.return_value = mock_response

        result = router.openai_decide(
            stage="model_selection",
            prompt="Which model should we use for this classification task?"
        )

        assert result["decision"] == "Use RandomForest classifier"
        assert result["model"] == "gpt-4o-mini"
        assert "usage" in result
        assert result["usage"]["total_tokens"] == 30

    def test_openai_decide_with_context(self, router):
        """Test decision with context."""
        mock_response = ChatCompletion(
            id="chatcmpl-test456",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Based on the data, use LogisticRegression",
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        )

        router.openai_client.chat.completions.create.return_value = mock_response

        context = {
            "n_samples": 150,
            "n_features": 4,
            "task": "classification"
        }

        result = router.openai_decide(
            stage="model_selection",
            prompt="Select a model",
            context=context
        )

        assert "decision" in result
        assert result["model"] == "gpt-4o-mini"

    def test_openai_decide_with_tools(self, router):
        """Test decision with function calling."""
        # Create a proper ChatCompletionMessageToolCall using dict
        from openai.types.chat import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function

        tool_call = ChatCompletionMessageToolCall(
            id="call_test123",
            type="function",
            function=Function(
                name="select_features",
                arguments='{"features": ["feature1", "feature2"]}'
            )
        )

        mock_response = ChatCompletion(
            id="chatcmpl-test789",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        )

        router.openai_client.chat.completions.create.return_value = mock_response

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "select_features",
                    "description": "Select features for modeling",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "features": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        ]

        result = router.openai_decide(
            stage="feature_selection",
            prompt="Select the best features",
            tools=tools
        )

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "select_features"
        assert result["tool_calls"][0]["arguments"]["features"] == ["feature1", "feature2"]

    def test_openai_decide_logs_decision(self, router, tmp_path):
        """Test that decisions are logged."""
        mock_response = ChatCompletion(
            id="chatcmpl-test999",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Decision made",
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )
        
        router.openai_client.chat.completions.create.return_value = mock_response
        
        result = router.openai_decide(
            stage="test_stage",
            prompt="Make a decision"
        )
        
        # Check that decision was logged
        log_file = router.decision_log.log_path
        assert log_file.exists()
        
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            last_entry = json.loads(lines[-1])
            assert last_entry["stage"] == "test_stage"
            assert last_entry["auth_model"] == "openai"

    def test_openai_decide_uses_low_temperature(self, router):
        """Test that decisions use low temperature for consistency."""
        mock_response = ChatCompletion(
            id="chatcmpl-temp",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Consistent decision",
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )
        
        router.openai_client.chat.completions.create.return_value = mock_response
        
        router.openai_decide(
            stage="test",
            prompt="Test prompt"
        )
        
        # Verify temperature was set to 0.1
        call_kwargs = router.openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.1


class TestOllamaDraft:
    """Test Ollama drafting functionality."""

    @pytest.fixture
    def router_with_ollama(self, tmp_path, monkeypatch):
        """Create a router with Ollama available."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI"):
            with patch("botds.llm.requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                
                decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
                config = {"openai_model": "gpt-4o-mini", "ollama_model": "llama3.2"}
                
                router = LLMRouter(config, decision_log)
                yield router

    def test_ollama_draft_when_available(self, router_with_ollama):
        """Test Ollama draft when service is available."""
        with patch("botds.llm.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": "This is a draft from Ollama"
            }
            
            result = router_with_ollama.ollama_draft(
                prompt="Draft a report summary"
            )
            
            assert result == "This is a draft from Ollama"
            mock_post.assert_called_once()

    def test_ollama_draft_when_unavailable(self, tmp_path, monkeypatch):
        """Test Ollama draft falls back when service is unavailable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI"):
            with patch("botds.llm.requests.get") as mock_get:
                mock_get.side_effect = Exception("Connection refused")
                
                decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
                config = {"openai_model": "gpt-4o-mini"}
                
                router = LLMRouter(config, decision_log)
                
                result = router.ollama_draft(prompt="Test prompt")

                # Should return None when unavailable
                assert result is None


class TestLLMRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self, tmp_path, monkeypatch):
        """Test handling of empty prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        with patch("botds.llm.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            decision_log = DecisionLog(tmp_path / "decision_log.jsonl")
            config = {"openai_model": "gpt-4o-mini"}
            
            router = LLMRouter(config, decision_log)
            router.openai_client = mock_client
            
            mock_response = ChatCompletion(
                id="chatcmpl-empty",
                object="chat.completion",
                created=1234567890,
                model="gpt-4o-mini",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            
            mock_client.chat.completions.create.return_value = mock_response
            
            result = router.openai_decide(stage="test", prompt="")
            
            assert "decision" in result

