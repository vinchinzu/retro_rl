"""``python -m super_metroid.generalist`` — status, diagnose, or train."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--status", "status", "corpus"}:
        from super_metroid.generalist.corpus import main as corpus_main

        rest = args[1:] if args and args[0] in {"status", "corpus"} else args
        return corpus_main(["--status", *rest] if "--status" not in rest else rest)
    if args[0] == "train":
        from super_metroid.generalist.train import main as train_main

        return train_main(args[1:])
    if args[0] == "overnight":
        from super_metroid.generalist.overnight import main as overnight_main

        return overnight_main(args[1:])
    if args[0] in {"diagnose", "probe"}:
        from super_metroid.generalist.diagnose import main as diagnose_main

        return diagnose_main(args[1:])
    from super_metroid.generalist.train import main as train_main

    return train_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
