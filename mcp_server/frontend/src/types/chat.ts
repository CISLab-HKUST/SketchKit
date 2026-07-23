export type MessageRole = "user" | "assistant";
export type MessageStatus = "pending" | "processing" | "done" | "error";

/**
 * We normalize ALL assistant outputs into a single downloadable file.
 * - Input can include multiple files
 * - Output is always ONE file per request
 */
export type ResultType = "file";

export type OutputKind = "json" | "image" | "svg" | "glb" | "gif" | "unknown";

export interface ResultData {
  type: ResultType; // always "file"
  kind: OutputKind; // drives preview UI
  path: string; // backend-returned path (relative or absolute)
  file_name?: string; // optional; fallback to name parsed from path
  mime_type?: string; // optional; helps when extension is not reliable
  preview_url?: string; // data:image/png;base64,...
  metadata?: Record<string, any>;
}

export interface MessageContent {
  text?: string;
  statusText?: string; // Dynamic status from server

  // For user messages: previews / selections
  images?: string[]; // base64 data URLs for preview in chat
  objFiles?: string[]; // file names
  jsonFiles?: string[]; // file names

  // For assistant messages: single output file
  result?: ResultData;
  intermediate_results?: ResultData[];

  confidence?: number;
  parameters?: Record<string, any>;
  error?: string;
}

export interface Message {
  id: string;
  role: MessageRole;
  content: MessageContent;
  timestamp: Date;
  status: MessageStatus;
}

export interface ChatRequest {
  message: string;
  images?: File[];
  objFiles?: File[];
  jsonFiles?: File[];
  context?: string[];
}

export interface ChatResponse {
  reply: string;
  generated_file?: string;

  processed_files?: string[];
  generated_files?: string[];

  file_name?: string;
  mime_type?: string;

  error?: string;
}
