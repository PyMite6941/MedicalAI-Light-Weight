# For functionality
from optimum.onnxruntime import ORTModelForVision2Seq
# For styling
import questionary
from rich.console import Console
from rich.panel import Panel

console = Console()

def vision_main():
    model = ORTModelForVision2Seq.from_pretrained("./blip-image-captioning-base",provider="CPUExecutionProvider",local_files_only=True)

def symptom_check_main():
    console.print("[bold yellow]Symptom Check functionality is not yet implemented.[/bold yellow]")

def main():
    console.print(Panel.fit("[bold cyan]Welcome to MedicalAI - Light Weight[/bold cyan]!"))
    choice = questionary.select(
        "Which functionality would you like to use?",
        choices=[
            "Vision",
            "Symptom Check",
            "Exit",
        ],
        pointer=">",
    ).ask()
    if choice == "Vision":
        vision_main()
    elif choice == "Symptom Check":
        symptom_check_main()
    else:
        console.print("[bold red]Exiting...[/bold red]")
        exit()