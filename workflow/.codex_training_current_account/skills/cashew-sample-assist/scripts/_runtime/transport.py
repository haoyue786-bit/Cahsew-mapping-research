"""Send structured skill input and typed images to the current Codex account."""
import base64
import importlib.util
import json
from pathlib import Path
import queue
import sys
import tempfile
import time

import jsonschema
from PIL import Image

from .common import SKILL_ROOT, SKILL, MODEL, EFFORT, SERVICE_TIER, PROTOCOL_ROOT, read, write, digest

from .hosted_transport import AppServer, DISABLED_FEATURES

ROOT = None


def binding_verifier():
    path = SKILL.parent/'scripts/verify_explicit_skill_binding.py'
    spec = importlib.util.spec_from_file_location('slim128_binding', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_image(record):
    path = Path(record['path']).resolve()
    allowed = False
    for root in (ROOT, SKILL_ROOT/'assets'):
        try:
            path.relative_to(root.resolve())
            allowed = True
        except ValueError:
            pass
    if not allowed:
        raise ValueError('Image outside declared new-run/reference assets')
    with Image.open(path) as im:
        fmt, size = im.format, im.size
        im.verify()
    if fmt not in ('PNG','JPEG') or min(size)<64 or max(size)>2048:
        raise ValueError('Every delivered image must be a real PNG/JPEG between64 and2048 display pixels')
    if record.get('sha256') and digest(path)!=record['sha256']:
        raise ValueError('Registered/prepared image checksum changed')
    result = dict(record, path=str(path), width=size[0], height=size[1], sha256=digest(path),
                  mime_type='image/png' if fmt == 'PNG' else 'image/jpeg')
    return result


def visible_metadata(record):
    """Keep long audit paths/hashes on host, not in the inference payload."""
    keys = ('image_id','reference_id','frame_id','candidate_id','provider','role','width','height',
            'source_years','source_date','annotation_allowed','bounds_wgs84','display_note',
            'label_scope','actual_AEF_grid','attribution','navigation','extent_wgs84','extent_precision',
            'unit_id','revision','coordinate_extent','original_AEF_pixel_edge_extent','requested_AEF_year','date_alignment_note','sam_source_frame_id',
            'native_pixel_edge_extent','display_to_native_scale','AOI_outline','date_unknown_or_mixed',
            'source_years_are_observed_endpoints_only','historical_selection','date_precision',
            'date_qualified_native_coverage_fraction','known_label_scope','label_scope_note','country','class_id','reference_scope')
    return {k:record[k] for k in keys if k in record}


def initial_image_parts(record):
    item = checked_image(record)
    url = 'data:'+item['mime_type']+';base64,'+base64.b64encode(Path(item['path']).read_bytes()).decode()
    return item, [{'type':'text','text':json.dumps(visible_metadata(item),ensure_ascii=False)},
                  {'type':'image','url':url,'detail':'original'}]


class ImageBudget:
    def __init__(self, records=(), maximum=32):
        self.maximum = maximum
        self.records = {}
        self.image_ids_by_hash = {}
        self.delivered = 0
        for record in records:
            self.register(record)

    def has_sam_review(self):
        return any('SAM_geometry_overlay' in r.get('role','') for r in self.records.values())

    def source_count(self):
        unique = [self.records[i] for i in self.image_ids_by_hash.values()]
        return sum(r.get('provider') in ('Google','Esri') and not r.get('reference_id')
                   and 'SAM_geometry' not in r.get('role','') for r in unique)

    def register(self, record):
        item = checked_image(record)
        if item['image_id'] in self.records:
            existing = self.records[item['image_id']]
            if existing['sha256'] != item['sha256']:
                raise ValueError('Image identity must be immutable')
            return existing
        if item['sha256'] in self.image_ids_by_hash:
            self.records[item['image_id']] = item
            return item
        if self.delivered >= self.maximum:
            return None
        self.records[item['image_id']] = item
        self.image_ids_by_hash[item['sha256']] = item['image_id']
        self.delivered += 1
        return item

    def receipt(self):
        return {'maximum_image_deliveries': self.maximum,'actual_image_deliveries':self.delivered,
                'source_view_count':self.source_count(),
                'SAM_review_overlay_delivered':self.has_sam_review(),'first_SAM_review_slot_reserved':False,
                'remaining_image_deliveries':self.maximum-self.delivered,
                'duplicate_images_redelivered':False,
                'image_aliases':len(self.records)-self.delivered,
                'delivered_dimensions':[[self.records[i]['width'],self.records[i]['height']]
                                        for i in self.image_ids_by_hash.values()]}


def image_tool_block_reason(tool, arguments, budget):
    if budget.delivered >= budget.maximum and tool not in ('submit_annotation', 'lookup_country_context'):
        return 'Image review budget reached; use the already reviewed units.'
    return None


def tool_content(result, budget):
    items, manifests = [], []
    metadata = {k:v for k,v in result.items() if k not in ('images','image_paths','image_png')}
    for record in result.get('images',[]):
        previous_count = budget.delivered
        item = budget.register(record)
        if item is None:
            metadata['image_delivery_note'] = 'Image already delivered or total display budget reached; use the existing images.'
            continue
        if budget.delivered == previous_count:
            metadata.setdefault('reused_images', []).append({
                'requested_image_id': item['image_id'],
                'previously_delivered_image_id': budget.image_ids_by_hash[item['sha256']],
            })
            continue
        manifests.append(item)
        url = 'data:'+item['mime_type']+';base64,'+base64.b64encode(Path(item['path']).read_bytes()).decode()
        items += [{'type':'inputText','text':json.dumps(visible_metadata(item),ensure_ascii=False)},
                  {'type':'inputImage','imageUrl':url}]
    metadata['case_image_budget'] = budget.receipt()
    items.insert(0, {'type':'inputText','text':json.dumps(metadata,ensure_ascii=False)})
    response = {'success':bool(result.get('success',True)),'contentItems':items}
    jsonschema.validate(response, read(PROTOCOL_ROOT/'DynamicToolCallResponse.json'))
    return response, manifests


def prepare_thread(developer, tools, model=MODEL):
    thread = {'model':model,'approvalPolicy':'never','baseInstructions':SKILL.read_text(),
              'developerInstructions':developer,'cwd':tempfile.mkdtemp(prefix='cashew_slim128_'),
              'runtimeWorkspaceRoots':[],'environments':[],
              'dynamicTools':[dict(t,type='function') for t in tools],
              'allowProviderModelFallback':False,
              'config':dict({'features.'+x:False for x in DISABLED_FEATURES},
                            model_reasoning_effort=EFFORT, web_search='disabled',project_doc_max_bytes=0)}
    if model == MODEL:
        thread['serviceTier'] = SERVICE_TIER
    jsonschema.validate(thread, read(PROTOCOL_ROOT/'v2/ThreadStartParams.json'))
    return thread


def prepare_turn(thread, directory, prompt, images, metadata, output_schema, prior_images=()):
    directory = Path(directory)
    if (directory/'status.json').exists():
        raise ValueError('Prepared/submitted turn exists; preserve its identity')
    retained = list(prior_images)
    # A follow-up must have slots for its direct inputs even when the earlier tool returns filled the budget.
    budget = ImageBudget(retained, maximum=max(32, len(retained) + len(images)))
    prior_count = budget.delivered
    inputs = [{'type':'skill','name':'cashew-sample-assist','path':str(SKILL)},
              {'type':'text','text':json.dumps({'prompt':prompt,'metadata':metadata},ensure_ascii=False)}]
    for record in images:
        previous_count = budget.delivered
        item = budget.register(record)
        if item is None:
            continue
        if budget.delivered == previous_count:
            inputs.append({'type':'text','text':json.dumps(dict(
                visible_metadata(item), previously_delivered_image_id=budget.image_ids_by_hash[item['sha256']]),
                ensure_ascii=False)})
            continue
        _, parts = initial_image_parts(item)
        inputs.extend(parts)
    turn = {'threadId':'SAME_THREAD_NOT_STARTED','model':thread['model'],'effort':EFFORT,
            'environments':[],'runtimeWorkspaceRoots':[],'input':inputs,'outputSchema':output_schema}
    write(directory/'turn_delivery_receipt.json',{'prior_case_images_retained':prior_count,'new_image_deliveries':budget.delivered-prior_count,'previous_images_retransmitted':False})
    receipt = binding_verifier().validate_binding(thread,turn,thread['model'],EFFORT)
    write(directory/'turn_request.prepared.json',turn)
    write(directory/'binding_receipt.json',receipt)
    write(directory/'image_registry.json',budget.records)
    write(directory/'image_budget.json',budget.receipt())
    write(directory/'status.json',{'status':'prepared','model_calls':0,'prepared_epoch':time.time()})
    return turn


class Session:
    def __init__(self,directory,thread,tool_module):
        self.directory = Path(directory)
        self.thread = thread
        self.tool_module = tool_module
        self.server = None
        self.thread_id = None
        self.service_tier = None
        self.tools = {x['name']:x for x in thread['dynamicTools']}
        self.last_token_usage = None
        self.tool_number = 0

    def _event(self, path, event):
        with Path(path).open('a') as handle:
            handle.write(json.dumps(event,ensure_ascii=False)+'\n')

    def request_and_wait(self,method,params):
        identity = self.server.request(method,params)
        start = time.time()
        while True:
            if time.time()-start > 300:
                raise TimeoutError('RPC unresolved: '+method+'; do not replay')
            try:
                event = self.server.events.get(timeout=15)
            except queue.Empty:
                continue
            self._event(self.directory/'session_events.jsonl',event)
            if event.get('transport_error'):
                raise RuntimeError(event['transport_error'])
            if event.get('id') == identity and 'method' not in event:
                if event.get('error'):
                    raise RuntimeError(event['error'])
                return event['result']

    def start(self):
        write(self.directory/'thread_request.json',self.thread)
        self.server = AppServer(self.directory)
        write(self.directory/'owned_process.json',{'app_server_pid':self.server.proc.pid,'started_epoch':time.time()})
        self.request_and_wait('initialize',{'clientInfo':{'name':'cashew_slim128','version':'10'},
                                          'capabilities':{'experimentalApi':True}})
        self.server.send({'method':'initialized','params':{}})
        actual = self.request_and_wait('thread/start',self.thread)
        if actual['model'] != self.thread['model']:
            raise ValueError('Model reroute rejected')
        if actual.get('reasoningEffort') != EFFORT:
            raise ValueError('App-server reasoning effort differs from the configured effort')
        self.thread_id = actual['thread']['id']
        self.service_tier = actual.get('serviceTier')
        write(self.directory/'thread_start_response.json',actual)
        return self.thread_id

    def compact_before_case(self, expected_input_tokens=12000):
        """Reserve room before the next case, not only after a near-full turn."""
        usage = (self.last_token_usage or {}).get('tokenUsage',{})
        window = usage.get('modelContextWindow')
        last = usage.get('last',{})
        used = last.get('inputTokens',0)+last.get('outputTokens',0)
        needed = bool(window and used+expected_input_tokens >= .75*window)
        receipt = {'thread_id':self.thread_id,'before_case':True,'reported_used_tokens':used,
                   'expected_next_case_input_tokens':expected_input_tokens,
                   'reported_context_window':window,'compaction_needed':needed,
                   'recorded_epoch':time.time()}
        write(self.directory/'context_budget_latest.json',receipt)
        if not needed:
            return receipt
        identity = self.server.request('thread/compact/start',{'threadId':self.thread_id})
        done, acknowledged, start = False, False, time.time()
        while not (done and acknowledged):
            if time.time()-start > 900:
                raise TimeoutError('Same-thread compaction unresolved; preserve identity')
            try:
                event = self.server.events.get(timeout=15)
            except queue.Empty:
                continue
            self._event(self.directory/'compaction_events.jsonl',event)
            if event.get('transport_error'):
                raise RuntimeError(event['transport_error'])
            if event.get('id') == identity and 'method' not in event:
                if event.get('error'):
                    raise RuntimeError(event['error'])
                acknowledged = True
            method, params = event.get('method'), event.get('params',{})
            if method == 'thread/compacted' or (method == 'item/completed' and params.get('item',{}).get('type') == 'contextCompaction'):
                done = True
            if method == 'thread/tokenUsage/updated':
                self.last_token_usage = params
        receipt.update(compacted_same_thread=True,completed_epoch=time.time())
        write(self.directory/('compaction_'+str(int(time.time()))+'.json'),receipt)
        return receipt

    def run_turn(self,directory,context,timeout_seconds=7200):
        directory = Path(directory)
        state = read(directory/'status.json')
        if state['status'] != 'prepared' or state['model_calls']:
            raise RuntimeError('Never retry a submitted/unresolved turn')
        turn = read(directory/'turn_request.prepared.json')
        turn['threadId'] = self.thread_id
        binding_verifier().validate_binding(self.thread,turn,self.thread['model'],EFFORT)
        saved_budget = read(directory/'image_budget.json')['maximum_image_deliveries']
        budget = ImageBudget(read(directory/'image_registry.json').values(), maximum=saved_budget)
        write(directory/'turn_request.json',turn)
        state.update(status='submitting',model_calls=1,thread_id=self.thread_id,started_epoch=time.time())
        write(directory/'status.json',state)
        request = self.server.request('turn/start',turn)
        state['status'] = 'submitted'
        write(directory/'status.json',state)
        texts, start = [], time.time()
        try:
            while True:
                if time.time()-start > timeout_seconds:
                    raise TimeoutError('Turn unresolved; preserve exact identity')
                try:
                    event = self.server.events.get(timeout=15)
                except queue.Empty:
                    continue
                self._event(directory/'events.jsonl',event)
                if event.get('transport_error'):
                    raise RuntimeError(event['transport_error'])
                if event.get('id') == request and 'method' not in event:
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    actual = event['result']['turn']
                    state.update(status='running',turn_id=actual['id'],server_turn_status=actual['status'])
                    write(directory/'turn_start_response.json',event['result'])
                    write(directory/'status.json',state)
                    continue
                method, params = event.get('method',''), event.get('params',{})
                if 'id' in event and method:
                    if method != 'item/tool/call' or params.get('tool') not in self.tools:
                        self.server.send({'id':event['id'],'error':{'code':-32601,'message':'Only declared current-case tools are available'}})
                        raise RuntimeError('Unexpected external/environment tool')
                    tool = params['tool']
                    arguments = params['arguments']
                    if isinstance(arguments,str):
                        arguments = json.loads(arguments)
                    self.tool_number += 1
                    call_dir = directory/'tool_calls'/str(self.tool_number).zfill(3)
                    write(call_dir/'request.json',params)
                    tool_start = time.time()
                    try:
                        jsonschema.validate(arguments,self.tools[tool]['inputSchema'])
                        blocked = image_tool_block_reason(tool,arguments,budget)
                        if blocked:
                            result = {'success':tool!='segment_with_sam','metadata':{'reason':blocked,'already_delivered_image_ids':list(budget.records)},'images':[]}
                        else:
                            result = self.tool_module.dispatch_tool(tool,arguments,dict(context,run_dir=str(self.directory),registered_frames=budget.records))
                    except Exception as exc:
                        result = {'success':False,'metadata':{'error':type(exc).__name__+': '+str(exc)},'images':[]}
                    write(call_dir/'tool_result.json',result)
                    response, delivered = tool_content(result,budget)
                    self.server.send({'id':event['id'],'result':response})
                    write(call_dir/'delivery_receipt.json',{'actual_app_server_response_sent':True,'tool':tool,
                          'started_epoch':tool_start,'response_sent_epoch':time.time(),
                          'protocol_inputImage_count':len(delivered),
                          'images':delivered,'image_budget':budget.receipt(),
                          'model_visual_attention_not_certified':True})
                    write(directory/'image_registry.json',budget.records)
                    write(directory/'image_budget.json',budget.receipt())
                    continue
                if method == 'model/rerouted':
                    raise RuntimeError('Model reroute rejected')
                if method == 'item/completed' and params.get('item',{}).get('type') == 'agentMessage':
                    texts.append(params['item'].get('text',''))
                if method == 'thread/tokenUsage/updated':
                    self.last_token_usage = params
                    write(directory/'token_usage.json',params)
                if method == 'turn/completed':
                    completed = params.get('turn',{})
                    if completed.get('id') and completed['id'] != state.get('turn_id'):
                        continue
                    if completed.get('status') != 'completed':
                        raise RuntimeError(str(completed))
                    if not texts:
                        texts = [x['text'] for x in completed.get('items',[]) if x.get('type')=='agentMessage' and x.get('text')]
                    value = json.loads(texts[-1])
                    write(directory/'model_result_raw.json',value)
                    jsonschema.validate(value,turn['outputSchema'])
                    verdict = self.tool_module.validate_final(dict(context,registered_frames=budget.records),value)
                    write(directory/'final_validation.json',verdict)
                    if not verdict['eligible']:
                        raise ValueError('Completed result failed geometry/evidence validation')
                    write(directory/'result.json',value)
                    state.update(status='completed',completed_epoch=time.time(),result_sha256=digest(directory/'result.json'))
                    write(directory/'status.json',state)
                    return value,budget.records
        except BaseException as exc:
            state.update(status='requires_reconciliation',error_type=type(exc).__name__,error=str(exc),updated_epoch=time.time())
            write(directory/'status.json',state)
            raise

    def close(self):
        if self.server:
            self.server.close()
