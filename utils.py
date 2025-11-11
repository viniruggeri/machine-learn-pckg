import sys
import time
import random
from rich.console import Console

console = Console()

COLORS = ["cyan", "magenta", "green", "yellow", "blue"]

def log(msg, delay=0.2):
    color = random.choice(COLORS)
    console.print(f"[{color}]{msg}[/{color}]")
    time.sleep(delay)

def separator():
    console.print("[bold white]" + "-" * 50 + "[/bold white]")

def loading(text="Processing", duration=2):
    for i in range(duration):
        sys.stdout.write(f"\r{text}{'.' * (i % 4)}   ")
        sys.stdout.flush()
        time.sleep(0.5)
    print()

def success(msg):
    console.print(f"[bold green]✔ {msg}[/bold green]")

def fail(msg):
    console.print(f"[bold red]✖ {msg}[/bold red]")
