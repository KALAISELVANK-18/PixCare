import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'Details2.dart';

class Report extends StatefulWidget {
  final String? user;
  const Report({super.key,required this.user});

  @override
  State<Report> createState() => _ReportState();
}

class _ReportState extends State<Report> {
  String formatTimestamp(int timestamp) {
    var format = new DateFormat('dd-MMM-yyyy');
    var date = new DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    return format.format(date);
  }
 // Replace "default_value" with the actual default value you want to use


  late String? userType;
  @override
  void initState() {
    super.initState();
    super.initState();
    getUserType();

    @override
    void dispose() {
      super.dispose();
    }
  }
  Future<String?> getUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      userType = prefs.getString('userType');
    });
    return null;
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(

      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.only(top: 40),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              if(userType!=null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 0),
                child: StreamBuilder<QuerySnapshot>(
                    stream: FirebaseFirestore.instance
                        .collection(userType!)
                        .doc(FirebaseAuth.instance.currentUser!.email.toString())
                        .collection("posts")
                        .orderBy('time', descending: true)
                        .snapshots(),
                    builder: (context, AsyncSnapshot<QuerySnapshot> snapshot) {
                      if (snapshot.hasData && snapshot.data != null) {
                        List<DocumentSnapshot> nn = snapshot.data!.docs;
                        return SingleChildScrollView(
                          child: Padding(
                            padding: EdgeInsets.symmetric(horizontal: 10.0),
                            child: Column(
                              children: [
                                (nn.length == 0)
                                    ? Center(
                                        child: Padding(
                                          padding: const EdgeInsets.all(20.0),
                                          child: Text("No report found"),
                                        ),
                                      )
                                    : SizedBox(),
                                Column(
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Text("Reports",
                                          style: GoogleFonts.poppins(
                                              textStyle: TextStyle(
                                                  fontWeight: FontWeight.bold,
                                                  fontSize: 24))),
                                    ),
                                    ListView.builder(
                                      shrinkWrap: true,
                                      physics: NeverScrollableScrollPhysics(),
                                      itemCount: nn.length,
                                      itemBuilder:
                                          (BuildContext context, int index) {
                                        return Padding(
                                          padding: const EdgeInsets.all(1.0),
                                          child: Container(
                                            child: TextButton(
                                              onPressed: () {
                                                Navigator.of(context).push(
                                                    MaterialPageRoute(
                                                        builder: (context) =>
                                                            Details2(
                                                                imagePath: nn[
                                                                        index][
                                                                    "picture_url"],
                                                                result: nn,
                                                                index: index,
                                                                usertype: widget.user
                                                            )));
                                              },
                                              child: Container(
                                                decoration: BoxDecoration(
                                                    color: Color.fromRGBO(
                                                        198, 199, 203, 1),
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                            10)),
                                                child: Row(
                                                  mainAxisAlignment:
                                                      MainAxisAlignment.start,
                                                  children: [
                                                    Padding(
                                                      padding:
                                                          const EdgeInsets.all(
                                                              8.0),
                                                      child: Column(
                                                        children: [
                                                          Padding(
                                                            padding:
                                                                const EdgeInsets
                                                                    .all(1.0),
                                                            child: Container(
                                                                height: 110,
                                                                width: 110,
                                                                decoration:
                                                                    BoxDecoration(
                                                                  borderRadius:
                                                                      BorderRadius
                                                                          .circular(
                                                                              10),
                                                                  image:
                                                                      DecorationImage(
                                                                    fit: BoxFit
                                                                        .cover,
                                                                    image:
                                                                        NetworkImage(
                                                                      nn[index][
                                                                          "picture_url"],
                                                                      //"https://images.unsplash.com/photo-1566438480900-0609be27a4be?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8aW1hZ2V8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=600&q=60"
                                                                    ),
                                                                  ),
                                                                ),
                                                                child:
                                                                    SizedBox()),
                                                          ),
                                                        ],
                                                      ),
                                                    ),
                                                    Padding(
                                                      padding:
                                                          const EdgeInsets.only(
                                                              right: 0),
                                                      child: Column(
                                                        mainAxisAlignment:
                                                            MainAxisAlignment
                                                                .spaceBetween,
                                                        crossAxisAlignment:
                                                            CrossAxisAlignment
                                                                .start,
                                                        children: [
                                                          Container(
                                                              width: 180,
                                                              child: Padding(
                                                                padding: const EdgeInsets.only(left: 2),
                                                                child: Text(
                                                                    (nn[index][
                                                                            "diseases"])
                                                                        ? nn[index]
                                                                            [
                                                                            "title"]
                                                                        : "No pathology found",
                                                                    textAlign:
                                                                        TextAlign
                                                                            .start,
                                                                    style: GoogleFonts
                                                                        .poppins(
                                                                            textStyle:
                                                                                TextStyle(
                                                                      fontSize:
                                                                          20,
                                                                      color: Colors
                                                                          .black,
                                                                    ))),
                                                              )),
                                                          Row(
                                                            children: [
                                                              Icon(
                                                                Icons
                                                                    .calendar_month,
                                                                color: Colors
                                                                    .black,
                                                              ),
                                                              Container(
                                                                  width: 180,
                                                                  child: Text(
                                                                      formatTimestamp(nn[index]
                                                                              [
                                                                              "time"]
                                                                          .seconds),
                                                                      style: GoogleFonts
                                                                          .poppins(
                                                                              textStyle: TextStyle(
                                                                        fontSize:
                                                                            18,
                                                                        color: Colors
                                                                            .black,
                                                                        fontWeight:
                                                                            FontWeight.bold,
                                                                      )))),
                                                            ],
                                                          )
                                                        ],
                                                      ),
                                                    )
                                                  ],
                                                ),
                                              ),
                                            ),
                                          ),
                                        );
                                      },
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      } else {
                        return CircularProgressIndicator();
                      }
                    }),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
