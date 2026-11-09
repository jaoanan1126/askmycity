import { Source } from '@/lib/types';
import { FileText } from 'lucide-react';

interface SourcesListProps {
  sources: Source[];
}

export default function SourcesList({ sources }: SourcesListProps) {
  return (
    <div className="mt-2 p-3 bg-blue-50 rounded-xl text-sm">
      <div className="font-semibold text-blue-700 flex items-center gap-2 mb-2">
        <FileText className="w-4 h-4" />
        Sources
      </div>
      <div className="space-y-1">
        {sources.map((source, idx) => {
          const name = source.title
            ? source.section
              ? `${source.title} › ${source.section}`
              : source.title
            : source.source;
          return (
            <div key={idx} className="text-gray-700 text-xs">
              📄{' '}
              {source.url ? (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-700 underline hover:text-blue-900"
                >
                  {name}
                </a>
              ) : (
                name
              )}{' '}
              ({source.category}) - {Math.round(source.relevance_score * 100)}% relevant
            </div>
          );
        })}
      </div>
    </div>
  );
}