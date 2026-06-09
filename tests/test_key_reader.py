from pathlib import Path

from src.collector.key_reader import read_credentials, redact_text


def test_read_credentials_from_natural_language(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_EMAIL", raising=False)
    monkeypatch.delenv("GOOGLE_PASSWORD", raising=False)
    key_file = tmp_path / "key.md"
    key_file.write_text(
        "账号: student@example.com\n密码: super-secret\n链接: https://play.google.com/console/developers/abc\n",
        encoding="utf-8",
    )
    bundle = read_credentials(key_file)
    assert bundle.email == "student@example.com"
    assert bundle.password == "super-secret"
    assert bundle.console_url.startswith("https://play.google.com/console")
    assert bundle.redacted()["password"] == "<redacted>"


def test_redact_text_masks_email_and_password():
    text = "email: user@example.com\npassword: my-password"
    redacted = redact_text(text)
    assert "user@example.com" not in redacted
    assert "my-password" not in redacted
