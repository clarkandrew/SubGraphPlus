# SubgraphRAG+ Frontend

A modern Next.js frontend for the SubgraphRAG+ knowledge graph question answering system, featuring real-time chat interface and interactive graph visualization.

## Features

- **💬 Interactive Chat Interface**: Real-time conversation with streaming responses
- **📊 Knowledge Graph Visualization**: Interactive D3.js-powered graph exploration  
- **📈 Analytics Dashboard**: Query performance and system metrics
- **🔍 Document Management**: Upload and manage knowledge base documents
- **⚡ Real-time Updates**: Server-sent events for live response streaming
- **🎨 Modern UI**: Built with Next.js 15, React 19, shadcn/ui, and Tailwind CSS

## Quick Start

### Prerequisites

- Node.js 18+ 
- SubgraphRAG+ backend running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Create environment configuration
cp env.example .env.local

# Update .env.local with your API configuration
# NEXT_PUBLIC_API_URL=http://localhost:8000
# NEXT_PUBLIC_API_KEY=your_api_key

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Architecture

### Key Components

- **Chat Interface** (`src/components/chat-support.tsx`): Real-time chat with SSE streaming
- **Graph Visualization** (`src/components/data-table.tsx`): Interactive knowledge graph browser
- **Navigation** (`src/components/app-sidebar.tsx`): Application navigation and user management
- **Analytics** (`src/components/chart-area-interactive.tsx`): Performance metrics and insights

### API Integration

The frontend integrates with the SubgraphRAG+ REST API:

- **POST /query**: Chat queries with streaming responses
- **POST /ingest**: Document upload and ingestion
- **GET /graph/browse**: Knowledge graph exploration
- **POST /feedback**: User feedback collection

### Real-time Features

- **Server-Sent Events (SSE)**: Live streaming of query responses
- **WebSocket Support**: Real-time graph updates (optional)
- **Progressive Loading**: Incremental content rendering

## Development

### Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

### Environment Variables

Create `.env.local` with your configuration:

```bash
# Required
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=your_api_key

# Optional
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_DEBUG=false
```

### Customization

The frontend uses shadcn/ui components and can be customized via:

- **Themes**: Update `src/app/globals.css` for color schemes
- **Components**: Modify components in `src/components/ui/`
- **Layout**: Update `src/app/layout.tsx` for global layout changes

## Deployment

### Vercel (Recommended)

```bash
npm run build
npx vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment-Specific Builds

For different environments, create multiple `.env` files:

- `.env.local` (development)
- `.env.staging` 
- `.env.production`

## Contributing

1. Follow the existing code style and component patterns
2. Use TypeScript for all new components
3. Add proper error handling and loading states
4. Test with the SubgraphRAG+ backend API
5. Update documentation for new features

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [shadcn/ui Components](https://ui.shadcn.com/)
- [SubgraphRAG+ API Documentation](../docs/api_reference.md)
