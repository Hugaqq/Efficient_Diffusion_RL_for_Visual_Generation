"""Pluggable notifier interface: send(text, key). Feishu uses argv, never shell text."""
import json

from .commands import run_command


class FeishuNotifier:
    def __init__(self, config, *, dry_run=False, runner=run_command):
        if config['identity'] not in ('user', 'bot') or not config['chat_id'].startswith('oc_'):
            raise ValueError('explicit Feishu chat_id (oc_...) and identity (user|bot) required')
        timeout = config.get('timeout_seconds', 30)
        if type(timeout) not in (int, float) or not 0 < timeout <= 120:
            raise ValueError('notifier timeout_seconds must be finite, >0 and <=120')
        self.config, self.dry_run, self.runner = config, dry_run, runner

    def argv(self, text, key):
        if not isinstance(text, str):
            raise ValueError('notification text must be a string')
        if text.lstrip().startswith(('@', '-')):
            text = '[H3] ' + text  # never CLI @file expansion or a leading option
        argv = [self.config.get('binary', 'lark-cli'), 'im', '+messages-send',
                '--as', self.config['identity'], '--chat-id', self.config['chat_id'],
                '--text', text, '--idempotency-key', key]
        if self.dry_run:
            argv.append('--dry-run')
        return argv

    def send(self, text, key):
        argv = self.argv(text, key)
        if self.dry_run:
            print(json.dumps(dict(notifier_argv=argv), ensure_ascii=False))
            return
        result = self.runner(argv, timeout=self.config.get('timeout_seconds', 30), max_bytes=65536)
        response = json.loads(result['stdout'])
        if not isinstance(response, dict) or response.get('ok') is not True:
            # Do not silently approve missing scopes, risk confirmations or change identity.
            raise RuntimeError('Feishu did not acknowledge ok=true')
