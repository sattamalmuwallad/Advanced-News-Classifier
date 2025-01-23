package uob.oop;

public class HtmlParser {
    /***
     * Extract the title of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the title if it's been found. Otherwise, return "Title not found!".
     */
    public static String getNewsTitle(String _htmlCode) {
        String titleTagOpen = "<title>";
        String titleTagClose = "</title>";

        int titleStart = _htmlCode.indexOf(titleTagOpen) + titleTagOpen.length();
        int titleEnd = _htmlCode.indexOf(titleTagClose);

        if (titleStart != -1 && titleEnd != -1 && titleEnd > titleStart) {
            String strFullTitle = _htmlCode.substring(titleStart, titleEnd);
            return strFullTitle.substring(0, strFullTitle.indexOf(" |"));
        }

        return "Title not found!";
    }

    /***
     * Extract the content of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the content if it's been found. Otherwise, return "Content not found!".
     */
    public static String getNewsContent(String _htmlCode) {
        String contentTagOpen = "\"articleBody\": \"";
        String contentTagClose = " \",\"mainEntityOfPage\":";

        int contentStart = _htmlCode.indexOf(contentTagOpen) + contentTagOpen.length();
        int contentEnd = _htmlCode.indexOf(contentTagClose);

        if (contentStart != -1 && contentEnd != -1 && contentEnd > contentStart) {
            return _htmlCode.substring(contentStart, contentEnd).toLowerCase();
        }

        return "Content not found!";
    }

    public static NewsArticles.DataType getDataType(String _htmlCode) {
        //TODO Task 3.1 - 1.5 Marks
        if (_htmlCode.contains("<datatype>")) {
            int start = _htmlCode.indexOf("<datatype>") + "<datatype>".length();
            int end = _htmlCode.indexOf("</datatype>");
            String type = _htmlCode.substring(start, end).trim();
            if (type.equalsIgnoreCase("Training"))
                return NewsArticles.DataType.Training;
            else
                return NewsArticles.DataType.Testing;
        } else
            return NewsArticles.DataType.Testing;
    }

    public static String getLabel(String _htmlCode) {
        //TODO Task 3.2 - 1.5 Marks
        if (_htmlCode.contains("<label>")) {
            int start = _htmlCode.indexOf("<label>") + "<label>".length();
            int end = _htmlCode.indexOf("</label>");
            String label = _htmlCode.substring(start, end).trim();
            return label;
        } else
            return "-1";
    }
}
