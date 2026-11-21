"""Candidate dynamic tool transport for bounded, same-run history access.

The history router uses this client for declared history-enabled jobs. Its
caller must authenticate the journal and the complete evidence path before
constructing it.
"""
import json, os
from pathlib import Path
from journal_history import fetch, KINDS

def valid_args(args, cutoff):
    if not isinstance(args,dict) or set(args)!={'start_epoch','end_epoch','kinds'}:return False
    start,end,kinds=args['start_epoch'],args['end_epoch'],args['kinds']
    return (type(start) is int and type(end) is int and 0<=start<=end<cutoff and end-start<=39
            and isinstance(kinds,list) and bool(kinds) and all(isinstance(k,str) for k in kinds)
            and len(kinds)==len(set(kinds)) and not (set(kinds)-KINDS))
from persistent_luna import PersistentLunaClient

SPEC = {'type':'function', 'name':'query_training_history',
        'description':'Query committed records of this exact group, including E0 initialization, training, controls, source and fixed unlabelled imagery, action, reasoning, review, queries, interventions, trials, module_detail, module_gradients, annotated_training (actual updates and before/after visual training fit), and candidate_details (includes final frozen-library fit during SELECT). Query a narrow range of at most 40 records. Missing measurements mean not collected. Target point evaluation stays sealed.',
        'inputSchema':{'type':'object', 'additionalProperties':False,
          'properties':{'start_epoch':{'type':'integer','minimum':0}, 'end_epoch':{'type':'integer','minimum':0},
             'kinds':{'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'string','enum':sorted(KINDS)}}},
          'required':['start_epoch','end_epoch','kinds']}}

class HistoryClient(PersistentLunaClient):
    def __init__(self, codex_binary, timeout_seconds, *, root, job_id, run_id, cutoff_epoch, audit_dir):
        super().__init__(codex_binary, timeout_seconds)
        self.history_root = Path(root); self.history_job_id = job_id; self.history_run_id = run_id
        self.history_cutoff = cutoff_epoch; self.history_audit = Path(audit_dir)
        self.history_thread = None; self.history_calls = 0; self.history_required = False
        self.trial_calls = 0; self.trial_request = None; self.detail_calls = 0
        self._query_progress = {}; self._query_interrupting = False

    def initialize(self):
        result,_=self._request('initialize',{'clientInfo':{'name':'cashew-history-controller','version':'1'},
                                             'capabilities':{'experimentalApi':True}})
        self._send({'jsonrpc':'2.0','method':'initialized','params':{}})
        return result

    def _read_line(self, deadline):
        while True:
            message = super()._read_line(deadline)
            if message.get('method') != 'item/tool/call' or 'id' not in message:
                return message
            params = message.get('params', {})
            if self._query_interrupting:
                self._send({'jsonrpc':'2.0','id':message['id'],'error':{'code':-32602,'message':'This turn is being interrupted after a non-progressing query loop.'}})
                continue
            if params.get('tool') == 'read_training_tool_result':
                self._detail_call(message)
                continue
            if params.get('tool') == 'run_matched_training_trial':
                self._trial_call(message)
                continue
            self.history_calls += 1
            if params.get('tool') != 'query_training_history' or params.get('threadId') != self.history_thread or params.get('namespace') not in (None, ''):
                raise RuntimeError('unexpected tool or thread')
            args = params.get('arguments')
            valid=valid_args(args,self.history_cutoff)
            if valid:
                result = fetch(self.history_root, job_id=self.history_job_id, run_id=self.history_run_id,
                               cutoff_epoch=self.history_cutoff, start_epoch=args['start_epoch'], end_epoch=args['end_epoch'], kinds=args['kinds'])
                # Recheck the complete outgoing payload, not merely the prompt:
                # point-derived aliases and nested target labels must never
                # reach the Agent even if a producer record is malformed.
                from online_boundary import verify_online
                verify_online(result)
                if self.history_required:
                    from online_boundary import isolate_history_text
                    result = isolate_history_text(result)
            else:
                result = {'error':'Invalid query. Use prior epochs, the advertised evidence kinds, and at most 40 epochs per request.'}
            from training_tool_output_v1 import delivery
            delivered = delivery(result, 'query_training_history', self.history_calls)
            record = {'request':message, 'result':result, 'delivered_result':delivered}
            self._annotate_progress(record)
            dest = self.history_audit / ('tool_call_%02d.json' % self.history_calls)
            with dest.open('x', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            self._reply_evidence(record, valid, delivered)

    def _detail_call(self, message):
        from training_tool_output_v1 import read, SPEC as detail_spec
        from complete_history_store import _write_immutable
        params = message['params']
        if params.get('threadId') != self.history_thread or params.get('namespace') not in (None, ''):
            raise RuntimeError('detail query belongs to another thread')
        self.detail_calls += 1
        result = read(self.history_audit, params.get('arguments'), self.history_thread, version=2)
        path = self.history_audit / ('detail_tool_call_%02d.json' % self.detail_calls)
        record = {'request':message, 'result':result, 'reader_version':2}
        self._annotate_progress(record)
        _write_immutable(path, (json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        self._reply_evidence(record, result.get('status')=='ok', result)

    def _annotate_progress(self, record):
        from query_progress_v1 import annotate
        tool = record['request']['params']['tool']
        previous, repeat = self._query_progress.get(tool, (None, 0))
        self._query_progress[tool] = annotate(record, previous, repeat)

    def _reply_evidence(self, record, success, delivered):
        message = record['request']
        if 'rpc_error' in record:
            self._send({'jsonrpc':'2.0','id':message['id'],'error':record['rpc_error']})
        else:
            self._send({'jsonrpc':'2.0','id':message['id'],'result':{'success':success,
                'contentItems':[{'type':'inputText','text':json.dumps(delivered,ensure_ascii=False,allow_nan=False)}]}})
        if record['query_progress']['repeat'] >= 4:
            from query_progress_v1 import interrupt_stalled
            interrupt_stalled(self, message['params'])

    def _trial_call(self, message):
        from paired_trial_v1 import invoke, spec
        params = message['params']
        if (self.trial_request is None or not spec(self.trial_request)
                or params.get('threadId') != self.history_thread or params.get('namespace') not in (None, '')):
            raise RuntimeError('unadvertised trial tool or thread')
        self.trial_calls += 1
        if self.trial_calls > 8: raise RuntimeError('trial tool call limit exceeded')
        try:
            from history_audit import verify as verify_history, HistoryQueryRequired
            turn_ids = {params.get('turnId')}
            for path in self.history_audit.glob('tool_call_*.json'):
                turn_ids.add(json.loads(path.read_text())['request']['params'].get('turnId'))
            try:
                verify_history(self.history_audit,root=self.history_root,job_id=self.history_job_id,
                    run_id=self.history_run_id,cutoff_epoch=self.history_cutoff,thread_id=self.history_thread,
                    require_history=True,allowed_turn_ids=sorted(x for x in turn_ids if x),selection_request=self.trial_request)
            except HistoryQueryRequired:
                result = {'status':'history_required','formal_state_adopted':False,
                          'message':'Retrieve E0 initialization and relevant earlier committed training before a trial.'}
            else:
                result = invoke(params.get('arguments'), self.trial_request, self.history_root, self.history_cutoff)
        except ValueError:
            result = {'status':'invalid_arguments', 'formal_state_adopted':False,
                      'message':'Use one mechanism, one or two effective bounded controls, integer discrete controls, an advertised step budget and the required hypothesis/response/falsification strings.'}
        dest = self.history_audit / ('trial_tool_call_%02d.json' % self.trial_calls)
        from complete_history_store import _write_immutable
        from training_tool_output_v1 import delivery
        delivered = delivery(result, 'run_matched_training_trial', self.trial_calls)
        _write_immutable(dest, (json.dumps({'request':message,'result':result,'delivered_result':delivered},ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        self._send({'jsonrpc':'2.0','id':message['id'],'result':{'success':result['status']!='invalid_arguments',
                    'contentItems':[{'type':'inputText','text':json.dumps(delivered,ensure_ascii=False,allow_nan=False)}]}})
