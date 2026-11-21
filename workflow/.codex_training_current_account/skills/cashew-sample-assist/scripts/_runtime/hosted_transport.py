"""Isolated app-server IO only; no old scientific prompts or states imported."""
import os,json,queue,threading,subprocess
from pathlib import Path
from .common import training_account_home
DISABLED_FEATURES=['shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins', 'memories', 'multi_agent', 'multi_agent_v2', 'browser_use', 'computer_use', 'image_generation', 'goals', 'skill_search', 'tool_suggest', 'in_app_browser', 'in_app_local_automation', 'sleep_tool']
class AppServer:
    def __init__(self,path):
        self.path=Path(path)
        self.events=queue.Queue()
        self.sequence=0
        self.stderr=(self.path/'app_server.stderr.log').open('w')
        env=dict(os.environ)
        env['CODEX_HOME']=str(training_account_home(env))
        proxy=env.get('CASHEW_CODEX_PROXY')
        if proxy:
            env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, ALL_PROXY=proxy)
        env['NO_PROXY']=','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1', 'localhost']))
        # Codex owns current-account authentication; no token is copied or logged.
        command=[env.get('CASHEW_CODEX_BIN') or 'codex','app-server','--listen','stdio://']
        for feature in DISABLED_FEATURES:
            command+=['--disable',feature]
        command+=['-c','web_search="disabled"','-c','project_doc_max_bytes=0']
        isolated_cwd=json.loads((self.path/'thread_request.json').read_text())['cwd']
        if not Path(isolated_cwd).is_dir() or any(Path(isolated_cwd).iterdir()):
            raise RuntimeError('App-server launch cwd must be an existing empty isolated directory')
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,bufsize=1,env=env,cwd=isolated_cwd)
        threading.Thread(target=self._reader,daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            try:self.events.put(json.loads(line))
            except ValueError:self.events.put({'transport_error':'non-JSON stdout from app-server'})
        self.events.put({'transport_error':'app-server stdout closed'})

    def send(self,value):
        self.proc.stdin.write(json.dumps(value,ensure_ascii=False)+'\n')
        self.proc.stdin.flush()

    def request(self,method,params):
        self.sequence+=1
        self.send({'id':self.sequence,'method':method,'params':params})
        return self.sequence

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill();self.proc.wait(timeout=10)
        self.stderr.close()
