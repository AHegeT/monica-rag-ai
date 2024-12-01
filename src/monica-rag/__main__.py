import click
from .api.client import MonicaAPIClient
from .models.rag import MonicaRAG
from .utils.text_processing import format_search_results
from .analytics.contact_analytics import ContactAnalytics
from .config import config

@click.group()
def cli():
    """Monica RAG CLI tool for semantic search and analytics."""
    pass

@cli.command()
def test():
    """Test if the CLI is working."""
    click.echo("CLI is working!")

@cli.command()
@click.option('--query', '-q', help='Search query', required=True)
@click.option('--top-k', '-k', default=3, help='Number of results to return')
def search(query: str, top_k: int):
    """Search contacts using semantic similarity."""
    # Initialize clients
    api_client = MonicaAPIClient(
        base_url=config.BASE_URL,
        token=config.API_TOKEN
    )
    
    # Initialize RAG
    rag = MonicaRAG(api_client)
    rag.initialize()
    
    # Perform search
    results = rag.query(query, top_k=top_k)
    
    # Display results
    click.echo(f"\nSearch results for: {query}")
    click.echo(format_search_results(results))

@cli.command()
def update():
    """Update embeddings for all contacts."""
    api_client = MonicaAPIClient(
        base_url=config.BASE_URL,
        token=config.API_TOKEN
    )
    
    rag = MonicaRAG(api_client)
    rag.update_embeddings()
    click.echo("Successfully updated embeddings")

@cli.group()
def analytics():
    """Contact analytics commands."""
    pass

@analytics.command()
@click.option('--field', '-f', help='Field to check (e.g., email, phone)', required=True)
def completion(field: str):
    """Check completion rate for a specific field."""
    api_client = MonicaAPIClient(
        base_url=config.BASE_URL,
        token=config.API_TOKEN
    )
    
    # Get all contacts
    click.echo("Fetching contacts...")
    contacts = api_client.get_contacts()
    
    # Get contact fields for all contacts
    click.echo("Fetching contact fields...")
    contact_fields = api_client.get_all_contacts_fields()
    
    # Initialize analytics with both contacts and their fields
    analytics = ContactAnalytics(contacts, contact_fields)
    
    # Get completion rate
    click.echo("Calculating completion rate...")
    stats = analytics.get_field_completion_rate(field)
    
    # Display results
    click.echo(f"\nCompletion statistics for field: {field}")
    click.echo(f"Completion rate: {stats['completion_rate']}%")
    click.echo(f"Completed: {stats['completed']}")
    click.echo(f"Missing: {stats['missing']}")
    click.echo(f"Total contacts: {stats['total']}")
    
    # Optionally show value distribution
    if stats['completed'] > 0:
        click.echo("\nValue distribution:")
        distribution = analytics.get_field_value_distribution(field)
        for value, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"{value}: {count}")

@analytics.command()
@click.option('--field', '-f', help='Field to analyze (e.g., job, company)', required=True)
def distribution(field: str):
    """Show value distribution for a specific field."""
    api_client = MonicaAPIClient(
        base_url=config.BASE_URL,
        token=config.API_TOKEN
    )
    
    contacts = api_client.get_contacts()
    analytics = ContactAnalytics(contacts)
    
    distribution = analytics.get_field_value_distribution(field)
    
    click.echo(f"\nValue distribution for field: {field}")
    for value, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        click.echo(f"{value}: {count}")

@analytics.command()
@click.argument('fields', nargs=-1, required=True)
def multi_completion(fields):
    """Check completion rates for multiple fields."""
    api_client = MonicaAPIClient(
        base_url=config.BASE_URL,
        token=config.API_TOKEN
    )
    
    contacts = api_client.get_contacts()
    analytics = ContactAnalytics(contacts)
    
    results = analytics.get_multiple_fields_completion(fields)
    
    click.echo("\nCompletion rates for multiple fields:")
    for field, stats in results.items():
        click.echo(f"\nField: {field}")
        click.echo(f"Completion rate: {stats['completion_rate']}%")
        click.echo(f"Completed: {stats['completed']}")
        click.echo(f"Missing: {stats['missing']}")

if __name__ == '__main__':
    cli()