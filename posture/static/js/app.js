// Entry point: build the shell, start polling, hand over.
import { store, start } from "./store.js";
import { AppShell } from "./components/app-shell.js";

document.body.append(AppShell(store));
start();
