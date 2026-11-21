"""Synthetic-only adapter for the *candidate* legacy automatic pool/train_epoch.

No production entry point. Verified callbacks and safe, bound real data/checkpoint
lineage are still required before use on real training. An interrupted epoch
must discard its in-memory mutations and resume from a complete checkpoint.
"""
from contextlib import contextmanager

class ActualEpochPool:
    """Count real completed G calls and successful automatic-pool exposure.

    The supplied legacy_pool is automatic_negative_production_v4.Pool.
    A trusted epoch caller binds the G optimizer below. A scheduled position
    is committed only once its complete epoch checkpoint is saved; an epoch
    exception creates no cursor transaction, even if some GPU updates ran.
    """
    def __init__(self, legacy_pool, *, cursor, asset_check):
        if type(cursor) is not int or not 0 <= cursor < 128 or legacy_pool.cursor != cursor:
            raise ValueError('wrong legacy cursor')
        if not callable(asset_check): raise ValueError('asset validator required')
        self.inner=legacy_pool; self.cursor=cursor; self.asset_check=asset_check
        self.started=False; self.bound=False; self.finished=False
        self.pending=None; self.calls=0; self.completed=0; self.completed_patch_occurrences=0
        self.completed_pixel_occurrences=0; self.completed_auxiliary_steps=0

    @property
    def schedule(self): return self.inner.schedule

    def begin_epoch(self, steps, config, seed):
        if self.started or not self.bound or type(steps) is not int or steps < 1:
            raise RuntimeError('unknown or repeated epoch start')
        if self.asset_check() is not True: raise ValueError('unverified or changed pool inputs')
        self.started=True; self.steps=steps
        self.inner.begin_epoch(steps, config, seed)
        self.frozen = not bool(self.inner.active_weight)
        # Legacy advances by planned iterations even when the pool was paused;
        # source-only repair / aged pool must preserve its cursor.
        if self.frozen: self.inner.next_cursor=self.cursor

    def step(self, step, gan, common, model, device, seed):
        if not self.started or self.finished or not self.bound or self.pending is not None or step != self.calls or step >= self.steps:
            raise RuntimeError('unknown, duplicate or out-of-order pool step')
        before_count=len(self.inner.exposed); before_pixels=self.inner.pixels
        value=self.inner.step(step,gan,common,model,device,seed)
        self.calls+=1
        self.pending={'step':step, 'active':value is not None and bool(self.inner.active_weight),
          'patches':len(self.inner.exposed)-before_count, 'pixels':self.inner.pixels-before_pixels}
        return value

    def record_gradients(self,*args,**kwargs):
        if not self.bound or self.pending is None or len(args)!=6 or kwargs or args[-1]!=self.pending['step'] or not self.pending['active']:
            raise RuntimeError('gradient probe outside the active pending step')
        return self.inner.record_gradients(*args)

    def _before_g_step(self):
        if not self.started or self.finished or self.pending is None or not self.bound:
            raise RuntimeError('G optimizer call outside pending pool step')

    def _after_g_step(self):
        row=self.pending; self.pending=None; self.completed+=1
        if row['active']:
            self.completed_auxiliary_steps+=1
            self.completed_patch_occurrences+=row['patches']
            self.completed_pixel_occurrences+=row['pixels']

    @contextmanager
    def bind_generator(self, optimizer):
        if self.bound or self.started or not callable(getattr(optimizer,'step',None)):
            raise RuntimeError('invalid optimizer binding')
        original=optimizer.step
        def tapped(*args,**kwargs):
            self._before_g_step(); result=original(*args,**kwargs); self._after_g_step(); return result
        optimizer.step=tapped; self.bound=True
        try: yield
        finally: optimizer.step=original; self.bound=False

    def summary(self):
        if not self.started or not self.bound or self.pending is not None or self.calls != self.steps or self.completed != self.steps:
            raise RuntimeError('partial epoch cannot yield committable exposure')
        if self.asset_check() is not True: raise ValueError('pool inputs changed during epoch')
        self.finished=True
        result=self.inner.summary()
        consumed=0 if self.frozen else min(self.completed,128-self.cursor)
        result.update(generation_start_cursor=self.cursor,generation_end_cursor=self.cursor+consumed,
          processed_contiguous_positions=consumed, completed_G_updates=self.completed,
          completed_G_auxiliary_steps=self.completed_auxiliary_steps,
          completed_G_patch_occurrences=self.completed_patch_occurrences,
          completed_G_pixel_occurrences=self.completed_pixel_occurrences,
          exposure_meaning='forward plus successful generator step, not label correctness or proof of useful gradient',
          cursor_meaning='positions committed only with epoch checkpoint')
        return result
