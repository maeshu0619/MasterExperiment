import getpass
import os
import smtplib
import socket
import subprocess
import traceback
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional


@dataclass
class MailConfig:
    enabled: bool = False
    to_addr: str = "maejima@uec.ac.jp"
    from_addr: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password_env: str = "MYNET_MAIL_PASSWORD"
    use_tls: bool = True
    timeout: float = 10.0
    sendmail_path: str = "/usr/sbin/sendmail"


class TrainingMailNotifier:
    def __init__(self, config: MailConfig, writer=None):
        self.config = config
        self.writer = writer

    @classmethod
    def from_args(cls, args, writer=None):
        config = MailConfig(
            enabled=bool(getattr(args, "mail_notify", False)),
            to_addr=str(getattr(args, "mail_to", "maejima@uec.ac.jp")),
            from_addr=str(getattr(args, "mail_from", "")),
            smtp_host=str(getattr(args, "mail_smtp_host", "")),
            smtp_port=int(getattr(args, "mail_smtp_port", 587)),
            smtp_user=str(getattr(args, "mail_smtp_user", "")),
            smtp_password_env=str(getattr(args, "mail_smtp_password_env", "MYNET_MAIL_PASSWORD")),
            use_tls=bool(getattr(args, "mail_use_tls", True)),
            timeout=float(getattr(args, "mail_timeout", 10.0)),
            sendmail_path=str(getattr(args, "mail_sendmail_path", "/usr/sbin/sendmail")),
        )
        return cls(config, writer=writer)

    def _write(self, message):
        if self.writer is not None and hasattr(self.writer, "write"):
            try:
                self.writer.write(message)
                if hasattr(self.writer, "flush"):
                    self.writer.flush()
            except Exception:
                pass

    def _from_addr(self):
        if self.config.from_addr:
            return self.config.from_addr
        host = socket.getfqdn() or socket.gethostname() or "localhost"
        return f"{getpass.getuser()}@{host}"

    def send(self, subject: str, body: str) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.to_addr:
            self._write("Mail notify skipped: recipient is empty.")
            return False

        msg = EmailMessage()
        msg["To"] = self.config.to_addr
        msg["From"] = self._from_addr()
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            if self.config.smtp_host:
                self._send_smtp(msg)
            else:
                self._send_sendmail(msg)
            self._write(f"Mail notify sent: {subject}")
            return True
        except Exception as exc:
            details = []
            for attr in ("stdout", "stderr"):
                value = getattr(exc, attr, None)
                if value:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    details.append(f"{attr}: {str(value).strip()}")
            detail_text = "" if not details else " " + " ".join(details)
            self._write(
                f"Mail notify failed: {type(exc).__name__}: {exc}.{detail_text} "
                "If the local sendmail service is not configured for external mail, "
                "set --mail_smtp_host/--mail_smtp_user and the password env var."
            )
            return False

    def _send_smtp(self, msg: EmailMessage):
        password = os.environ.get(self.config.smtp_password_env, "")
        with smtplib.SMTP(
            self.config.smtp_host,
            self.config.smtp_port,
            timeout=self.config.timeout,
        ) as smtp:
            if self.config.use_tls:
                smtp.starttls()
            if self.config.smtp_user:
                smtp.login(self.config.smtp_user, password)
            smtp.send_message(msg)

    def _send_sendmail(self, msg: EmailMessage):
        if not os.path.exists(self.config.sendmail_path):
            raise FileNotFoundError(
                f"sendmail not found: {self.config.sendmail_path}; set --mail_smtp_host or --mail_sendmail_path"
            )
        result = subprocess.run(
            [self.config.sendmail_path, "-t", "-oi"],
            input=msg.as_bytes(),
            check=True,
            timeout=self.config.timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stderr:
            self._write(result.stderr.decode("utf-8", errors="replace").strip())

    def training_started(self, start_date: str, log_path: Optional[str]):
        subject = "[myNet] Training started"
        backend = (
            f"SMTP {self.config.smtp_host}:{self.config.smtp_port}"
            if self.config.smtp_host
            else f"sendmail {self.config.sendmail_path}"
        )
        body = "\n".join(
            [
                "myNet training started.",
                f"Started at: {start_date}",
                f"Host: {socket.gethostname()}",
                f"Backend: {backend}",
                f"Log: {log_path or ''}",
            ]
        )
        return self.send(subject, body)

    def episode_finished(self, episode: int, total_episodes: int, loss_value: float, model_path: str, log_path: Optional[str]):
        subject = f"[myNet] Episode {episode}/{total_episodes} finished"
        body = "\n".join(
            [
                "myNet training episode finished.",
                f"Episode: {episode}/{total_episodes}",
                f"Average episode loss: {loss_value:.6f}",
                f"Checkpoint: {model_path}",
                f"Log: {log_path or ''}",
            ]
        )
        return self.send(subject, body)

    def training_finished(self, elapsed_sec: float, finish_date: str, best_loss: Optional[float], log_path: Optional[str]):
        subject = "[myNet] Training finished"
        body = "\n".join(
            [
                "myNet training finished.",
                f"Elapsed seconds: {elapsed_sec:.3f}",
                f"Finished at: {finish_date}",
                f"Best loss: {'' if best_loss is None else f'{best_loss:.6f}'}",
                f"Log: {log_path or ''}",
            ]
        )
        return self.send(subject, body)

    def training_error(self, exc: BaseException, log_path: Optional[str]):
        subject = "[myNet] Training stopped with error"
        body = "\n".join(
            [
                "myNet training stopped with an error.",
                f"Error: {type(exc).__name__}: {exc}",
                f"Log: {log_path or ''}",
                "",
                "Traceback:",
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            ]
        )
        return self.send(subject, body)
