# SubgraphRAG+ Setup Scripts

This directory contains utility scripts for setting up and managing your SubgraphRAG+ environment.

## Setup Scripts

### Development Environment

To set up a local development environment using Python virtualenv:

```bash
./bin/setup_dev.sh
```

This script:
- Creates a Python virtual environment
- Installs all development dependencies
- Sets up Neo4j using Docker (if available)
- Downloads pre-trained models
- Initializes the database schema
- Loads sample data
- Runs tests to verify the setup

Options:
- `--skip-tests`: Skip running tests
- `--skip-neo4j`: Skip Neo4j setup
- `--skip-sample-data`: Skip loading sample data
- `--python VERSION`: Use specific Python version

### Docker Environment

To set up a Docker-based environment:

```bash
./bin/setup_docker.sh
```

This script:
- Builds and starts all required Docker containers
- Creates necessary configuration files
- Loads sample data (if not skipped)

Options:
- `--skip-sample-data`: Skip loading sample data
- `--rebuild`: Force rebuild of Docker images
- `--pull`: Force pull latest base images
- `--foreground`: Run in foreground (not detached)

## Management Scripts

### Docker Management

```bash
./bin/docker-setup.sh [command]
```

Commands:
- `start`: Start all services
- `stop`: Stop all services
- `restart`: Restart all services
- `rebuild`: Rebuild and restart services
- `logs`: View container logs
- `status`: Show container status
- `resources`: Show resource usage
- `sample-data`: Load sample data
- `backup`: Create backup

### Running the Application

```bash
./bin/run.sh
```

Starts the application server with default settings.

### Running Tests

```bash
./bin/run_tests.sh
```

Options:
- `-t, --type TYPE`: Type of tests (unit, integration, all)
- `-c, --coverage`: Generate coverage report
- `-v, --verbose`: Verbose output

### Benchmarking

```bash
./bin/run_benchmark.sh
```

Options:
- `-i, --input FILE`: Input questions file
- `-o, --output FILE`: Output results file
- `-a, --adversarial`: Run adversarial benchmark

### Backup

```bash
./bin/backup.sh [action] [options]
```

Actions:
- `backup`: Create a new backup (default)
- `restore`: Restore from a backup
- `list`: List available backups

### Demo

```bash
./bin/demo.sh
```

Interactive demo script with pre-configured sample queries.

## Legacy Scripts

These scripts are maintained for backward compatibility but it's recommended to use the new setup scripts:

- `quickstart.sh`: Combined setup script (use `setup_dev.sh` instead)
- `setup.sh`: Basic setup (use `setup_dev.sh` instead)