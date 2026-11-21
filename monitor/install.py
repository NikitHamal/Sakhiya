"""Install the v6 mining dashboard cells into the live marimo notebook (run via the kernel; idempotent).
Wipes every cell that is not a current tts_* card (errored / stale / non-TTS), edits current cells in place, re-runs them."""
import sys
sys.path.insert(0, "/work/tts/repo")
import importlib.util as _u
_s = _u.spec_from_file_location("sakhiya_cells", "/work/tts/repo/monitor/cells.py"); C = _u.module_from_spec(_s); _s.loader.exec_module(C)
import marimo._code_mode as cm


async def main():
    async with cm.get_context(skip_staleness_check=True) as ctx:
        by_name = {c.name: c.id for c in ctx.cells.values()}
        for c in list(ctx.cells.values()):
            if c.name not in C.CELLS:
                ctx.delete_cell(c.id)
        ids = {}
        for name, src in C.CELLS.items():
            if name in by_name:
                ids[name] = by_name[name]; ctx.edit_cell(ids[name], src.strip(), hide_code=True)
            else:
                ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True)
        # keepalive first (defines mo/refresh/ui26), then the rest, so dependants never see a missing global
        for name in C.CELLS:
            ctx.run_cell(ids[name])
        print("updated", list(C.CELLS))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
# In the marimo kernel (top-level await): exec the file's source then `await main()` (see scripts/molab_install_cards.py).
