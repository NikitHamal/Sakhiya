"""Install the v10 MoLab cards (training box) into the live marimo notebook (run inside the kernel; idempotent).
Deletes every cell that is not a current card, edits current cells in place, re-runs them.
In the kernel: exec this file's source up to the main-guard, then await main() (top-level await)."""
import importlib.util as _u
_s = _u.spec_from_file_location("sakhiya_cells", "/work/tts/repo9/monitor/cells_v10.py"); C = _u.module_from_spec(_s); _s.loader.exec_module(C)
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
        for name in C.CELLS:
            ctx.run_cell(ids[name])
        print("cards:", list(C.CELLS))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
