$(function () {
    var getData = function (request, response) {
        $.getJSON(
            "_autocomp/" + request.term,
            function (data) {
                console.log(data)
                var array = $.map(data, function (m) {
                    return {
                        label: m.TitleArtist,
                        search: m.search,
                        songID: m.songID
                    };
                });
                response(array);
            });
    };

    var selectItem = function (event, ui) {
        console.log("Select item",ui);
        // event.preventDefault();
        $(this).text(ui.item.label);
        $("#songSearch").attr("value", ui.item.search)
        $("#songID").attr("value",ui.item.songID)
        // $('form[name=searchSong]').attr('action', ui.item.url);
        //window.open(ui.item.url);
        // return false;
    }

    $("#autocomplete").autocomplete({
        source: getData,
        select: selectItem,
        minLength: 4
    });
});

$(document).ready(function () {
    $(".progress").hide();
});

$("#search").click(function () {
    $(".progress").show();
});

