import json
from unittest.mock import Mock

import pytest

import chatcli


def test_default_config_path():
    assert chatcli.CONFIG_FILE == chatcli.CONFIG_DIR / "config" / "config.json"


def test_main_loads_and_saves_custom_config(tmp_path):
    config_path = tmp_path / "nested" / "custom.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"model": "custom-model"}), encoding="utf-8")

    result = chatcli.main(["--config", str(config_path), "--save-config"])

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result == 0
    assert saved_config["model"] == "custom-model"


def test_load_config_defaults_extra_body(monkeypatch, tmp_path):
    monkeypatch.setattr(chatcli, "CONFIG_FILE", tmp_path / "missing.json")

    config = chatcli.load_config()

    assert config["extra_body"] == {}


def test_load_config_accepts_extra_body_object(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"extra_body": {"thinking": {"type": "disabled"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(chatcli, "CONFIG_FILE", config_path)

    config = chatcli.load_config()

    assert config["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("invalid_value", [None, [], "invalid", 1])
def test_load_config_rejects_non_object_extra_body(
    monkeypatch, tmp_path, capsys, invalid_value
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"extra_body": invalid_value}), encoding="utf-8"
    )
    monkeypatch.setattr(chatcli, "CONFIG_FILE", config_path)

    config = chatcli.load_config()

    assert config["extra_body"] == {}
    assert "extra_body 必须是 JSON 对象" in capsys.readouterr().err


def test_save_and_load_chat_preserve_extra_body(monkeypatch, tmp_path):
    chat_path = tmp_path / "chat.json"
    app = chatcli.ChatApp.__new__(chatcli.ChatApp)
    app.console = Mock()
    app.model = "example-model"
    app.temperature = 0.4
    app.extra_body = {"vendor_option": True}
    app.messages = [{"role": "user", "content": "hello"}]

    app.save_chat(str(chat_path))

    saved_data = json.loads(chat_path.read_text(encoding="utf-8"))
    assert saved_data["extra_body"] == {"vendor_option": True}

    app.extra_body = {}
    app.persist_session_log = Mock()
    app.load_chat(str(chat_path))

    assert app.extra_body == {"vendor_option": True}
    app.persist_session_log.assert_called_once_with()


@pytest.mark.parametrize(
    ("extra_body", "expected_in_request"),
    [({}, False), ({"vendor_option": True}, True)],
)
def test_stream_reply_only_sends_nonempty_extra_body(
    extra_body, expected_in_request
):
    app = chatcli.ChatApp.__new__(chatcli.ChatApp)
    app.console = Mock()
    app.client = Mock()
    app.client.chat.completions.create.return_value = []
    app.model = "example-model"
    app.messages = []
    app.temperature = 0.7
    app.extra_body = extra_body

    reply, usage = app.stream_reply()

    request = app.client.chat.completions.create.call_args.kwargs
    assert ("extra_body" in request) is expected_in_request
    if expected_in_request:
        assert request["extra_body"] == extra_body
    assert reply is None
    assert usage is None