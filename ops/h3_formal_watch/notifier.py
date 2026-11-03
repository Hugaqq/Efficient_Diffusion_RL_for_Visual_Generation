"""Pluggable notifier interface: send(text, key). Feishu uses argv, never shell text."""
import json
import re

from .commands import run_command


def validate_target(config):
    """Fail closed: one bot-to-user DM, no chat/group or alternate target keys."""
    allowed = {'alert_user_open_id', 'identity', 'binary', 'timeout_seconds'}
    if (not isinstance(config, dict) or set(config) - allowed
            or config.get('identity') != 'bot'
            or not isinstance(config.get('alert_user_open_id'), str)
            or not re.fullmatch(r'ou_[A-Za-z0-9]{1,128}', config['alert_user_open_id'])):
        raise ValueError('alerts require identity=bot and one alert_user_open_id (ou_...); '
                         'chat/group/alternate targets are forbidden')


class FeishuNotifier:
    def __init__(self, config, *, dry_run=False, runner=run_command):
        validate_target(config)
        timeout = config.get('timeout_seconds', 30)
        if type(timeout) not in (int, float) or not 0 < timeout <= 120:
            raise ValueError('notifier timeout_seconds must be finite, >0 and <=120')
        self.config, self.dry_run, self.runner = config, dry_run, runner

    def argv(self, text, key):
        validate_target(self.config)  # recheck even if a caller mutated configuration
        if not isinstance(text, str):
            raise ValueError('notification text must be a string')
        if text.lstrip().startswith(('@', '-')):
            text = '[H3] ' + text  # never CLI @file expansion or a leading option
        argv = [self.config.get('binary', 'lark-cli'), 'im', '+messages-send',
                '--as', 'bot', '--user-id', self.config['alert_user_open_id'],
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
