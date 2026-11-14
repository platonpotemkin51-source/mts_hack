export class RequestError extends Error {
  constructor(public status: number) {
    super("API request failed");
  }
}
