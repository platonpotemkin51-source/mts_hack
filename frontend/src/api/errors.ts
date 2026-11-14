export class RequestError extends Error {
  constructor(public status: number, public code?: string) {
    super("API request failed");
  }
}
